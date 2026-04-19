"""Wire protocol helpers for pipeline stages.

Tensors go over HTTP as raw fp16 bytes. A small JSON header at the start of
every request carries shape/dtype + seq_id + position metadata. Simple enough
to port to Swift/Network.framework later without introducing msgpack or
protobuf dependencies.

Frame format on a single POST /forward:
  Content-Type: application/octet-stream
  Body = [4-byte big-endian header_len][utf8 JSON header][raw tensor bytes]

Header fields:
  seq_id:     str           session identifier
  stage_kind: "first"|"mid"|"last"
  shape:      [int,...]     tensor shape (ignored for first-stage, which sends tokens)
  dtype:      "float16"|"int32"
  positions:  [int,...]     absolute position ids for the T tokens being processed
  temperature, top_p, top_k: sampling params (only the last stage reads these)
"""
from __future__ import annotations
import json
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    seq_id: str
    stage_kind: str
    positions: list[int]
    tensor: np.ndarray
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    max_new_tokens: int = 1
    meta: dict = field(default_factory=dict)

    def encode(self) -> bytes:
        header = {
            "seq_id": self.seq_id,
            "stage_kind": self.stage_kind,
            "positions": self.positions,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "meta": self.meta,
        }
        hbytes = json.dumps(header).encode("utf-8")
        body = self.tensor.tobytes()
        return struct.pack(">I", len(hbytes)) + hbytes + body

    @classmethod
    def decode(cls, raw: bytes) -> "Frame":
        (hlen,) = struct.unpack(">I", raw[:4])
        header = json.loads(raw[4 : 4 + hlen].decode("utf-8"))
        tensor_bytes = raw[4 + hlen :]
        dtype = np.dtype(header["dtype"])
        arr = np.frombuffer(tensor_bytes, dtype=dtype).reshape(header["shape"]).copy()
        return cls(
            seq_id=header["seq_id"],
            stage_kind=header["stage_kind"],
            positions=header["positions"],
            tensor=arr,
            temperature=header.get("temperature", 0.7),
            top_p=header.get("top_p", 0.95),
            top_k=header.get("top_k", 40),
            max_new_tokens=header.get("max_new_tokens", 1),
            meta=header.get("meta", {}),
        )


@dataclass
class Response:
    """Worker response. For non-last stages carries hidden_state;
    for the last stage carries the sampled token id + logprob info."""
    seq_id: str
    shape: list[int]
    dtype: str
    tensor: bytes
    token_id: Optional[int] = None  # last-stage only
    eos: bool = False
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    def encode(self) -> bytes:
        header = {
            "seq_id": self.seq_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "token_id": self.token_id,
            "eos": self.eos,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
        }
        hbytes = json.dumps(header).encode("utf-8")
        return struct.pack(">I", len(hbytes)) + hbytes + self.tensor

    @classmethod
    def decode(cls, raw: bytes) -> "Response":
        (hlen,) = struct.unpack(">I", raw[:4])
        header = json.loads(raw[4 : 4 + hlen].decode("utf-8"))
        body = raw[4 + hlen :]
        return cls(
            seq_id=header["seq_id"],
            shape=header["shape"],
            dtype=header["dtype"],
            tensor=body,
            token_id=header.get("token_id"),
            eos=header.get("eos", False),
            latency_ms=header.get("latency_ms", 0.0),
            tokens_per_second=header.get("tokens_per_second", 0.0),
        )
