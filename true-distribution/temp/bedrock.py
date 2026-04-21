"""Tiny asyncio wrapper around Bedrock's streaming InvokeModel API.

Bedrock's SDK is blocking, so we run the `invoke_model_with_response_stream`
iteration in a thread and bridge deltas to the event loop via an asyncio.Queue.
Keeps the server.py coroutine happy.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

import boto3

# US cross-region inference profile — required to invoke Sonnet 4.6 on-demand
# from us-east-1 without hitting per-region capacity issues.
DEFAULT_MODEL_ID = os.environ.get(
    "REVIVE_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"
)
DEFAULT_REGION = os.environ.get("REVIVE_BEDROCK_REGION", "us-east-1")


class BedrockStreamer:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION):
        self.model_id = model_id
        self._client = boto3.client("bedrock-runtime", region_name=region)

    async def stream_text(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer() -> None:
            try:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                })
                resp = self._client.invoke_model_with_response_stream(
                    modelId=self.model_id,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                for event in resp["body"]:
                    raw = event.get("chunk", {}).get("bytes")
                    if not raw:
                        continue
                    data = json.loads(raw.decode("utf-8"))
                    if data.get("type") == "content_block_delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            loop.call_soon_threadsafe(q.put_nowait, text)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

        loop.run_in_executor(None, producer)

        while True:
            item = await q.get()
            if item is SENTINEL:
                return
            if isinstance(item, Exception):
                raise item
            yield item
