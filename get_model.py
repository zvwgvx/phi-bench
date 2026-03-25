import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars
from tqdm.auto import tqdm

MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
OUTPUT_DIR = Path("./models/llama-3.2-1b")
ENV_PATH = Path(".env")

def load_hf_token() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            if key.strip() == "HF_TOKEN":
                return value.strip().strip("'").strip('"')
    return os.getenv("HF_TOKEN", "")


def main() -> None:
    disable_progress_bars()
    token = load_hf_token()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with tqdm(total=1, desc="download", unit="model") as progress:
        snapshot_download(
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.tiktoken",
                "*.txt",
            ],
            repo_id=MODEL_ID,
            local_dir=OUTPUT_DIR,
            token=token,
        )
        progress.update(1)

if __name__ == "__main__":
    main()
