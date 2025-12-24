import argparse
from pathlib import Path

from huggingface_hub import HfApi


def upload_to_hub(adapter_path: str, repo_id: str):
    """Upload a LoRA adapter to HuggingFace Hub.

    Args:
        adapter_path: Path to the saved LoRA adapter directory
        repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
    """

    _adapter_path = Path(adapter_path)

    if not _adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {_adapter_path}")

    if _adapter_path.is_file():
        raise ValueError(f"Adapter path is a file: {adapter_path}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=_adapter_path,
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=["checkpoint-*"],
    )

    print(f"\nLoRA adapter uploaded to: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload a LoRA adapter to HuggingFace Hub"
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="./outputs",
        help="Path to the saved LoRA adapter",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/tiny-random-llama-lora')",
    )

    args = parser.parse_args()
    upload_to_hub(
        adapter_path=args.adapter_path,
        repo_id=args.repo_id,
    )


if __name__ == "__main__":
    main()
