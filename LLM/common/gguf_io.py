"""Thin wrappers around llama.cpp tooling and a round-trip loader."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LLAMA_DIR = REPO_ROOT / "llama.cpp"


@dataclass(frozen=True)
class LlamaPaths:
    root: Path

    @property
    def convert_script(self) -> Path:
        return self.root / "convert_hf_to_gguf.py"

    @property
    def quantize_bin(self) -> Path:
        for candidate in (
            self.root / "build" / "bin" / "llama-quantize",
            self.root / "build" / "bin" / "quantize",
        ):
            if candidate.exists():
                return candidate
        return self.root / "build" / "bin" / "llama-quantize"

    @property
    def imatrix_bin(self) -> Path:
        for candidate in (
            self.root / "build" / "bin" / "llama-imatrix",
            self.root / "build" / "bin" / "imatrix",
        ):
            if candidate.exists():
                return candidate
        return self.root / "build" / "bin" / "llama-imatrix"


def find_llama() -> LlamaPaths:
    env = os.environ.get("REVIVE_LLAMA_DIR")
    root = Path(env) if env else DEFAULT_LLAMA_DIR
    return LlamaPaths(root=root)


def check_binaries(paths: LlamaPaths | None = None) -> dict[str, bool]:
    paths = paths or find_llama()
    return {
        "llama.cpp dir": paths.root.exists(),
        "convert_hf_to_gguf.py": paths.convert_script.exists(),
        "quantize": paths.quantize_bin.exists(),
        "imatrix": paths.imatrix_bin.exists(),
    }


def convert_hf_to_gguf(hf_dir: Path, out_path: Path, outtype: str = "f16") -> None:
    paths = find_llama()
    if not paths.convert_script.exists():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found at {paths.convert_script}. "
            "Run macos/setup.sh or rpi/setup.sh to build llama.cpp first."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python3", str(paths.convert_script),
        str(hf_dir),
        "--outfile", str(out_path),
        "--outtype", outtype,
    ]
    subprocess.run(cmd, check=True)


def run_imatrix(
    fp16_gguf: Path,
    calibration_txt: Path,
    out_imatrix: Path,
    chunks: int = 100,
) -> None:
    paths = find_llama()
    if not paths.imatrix_bin.exists():
        raise FileNotFoundError(
            f"imatrix binary not found at {paths.imatrix_bin}. "
            "Rebuild llama.cpp with the llama-imatrix target enabled."
        )
    out_imatrix.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(paths.imatrix_bin),
        "-m", str(fp16_gguf),
        "-f", str(calibration_txt),
        "-o", str(out_imatrix),
        "--chunks", str(chunks),
    ]
    subprocess.run(cmd, check=True)


def quantize(
    fp16_gguf: Path,
    out_gguf: Path,
    quant_type: str,
    imatrix: Path | None = None,
) -> None:
    paths = find_llama()
    if not paths.quantize_bin.exists():
        raise FileNotFoundError(
            f"quantize binary not found at {paths.quantize_bin}. "
            "Run setup.sh to build llama.cpp."
        )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(paths.quantize_bin)]
    if imatrix is not None and imatrix.exists():
        cmd += ["--imatrix", str(imatrix)]
    cmd += [str(fp16_gguf), str(out_gguf), quant_type]
    subprocess.run(cmd, check=True)


def sanity_load(gguf_path: Path) -> bool:
    """Round-trip load via llama-cpp-python and run one forward pass."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[gguf_io] llama-cpp-python not installed; skipping sanity load")
        return False
    try:
        llm = Llama(model_path=str(gguf_path), n_ctx=512, verbose=False)
        out = llm("hello", max_tokens=4)
        del llm
        return bool(out)
    except Exception as exc:
        print(f"[gguf_io] sanity load failed: {exc}")
        return False


def gguf_name(role: str, size_label: str, tier: str, quant: str) -> str:
    return f"revive-{role}-qwen3-{size_label}-{tier}-{quant}.gguf"


def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    n = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"
