from pathlib import Path

from huggingface_hub import HfApi, create_repo


ADAPTER_FOLDER = Path("/Users/your_user_name/Downloads/qwen2_5_vl_lora_512Res/adapter_model")
REPO_ID = "quist99/book-spine-ocr-qwen25vl-512Res-adapter"
PRIVATE = False


def main() -> None:
    if not ADAPTER_FOLDER.exists() or not ADAPTER_FOLDER.is_dir():
        raise FileNotFoundError(f"Adapter folder not found: {ADAPTER_FOLDER}")

    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError("Not logged in to Hugging Face. Run: hf auth login") from exc

    create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

    commit = api.upload_folder(
        folder_path=str(ADAPTER_FOLDER),
        repo_id=REPO_ID,
        repo_type="model",
    )
    print(f"Uploaded adapter folder to {REPO_ID}. Commit: {commit.oid}")


if __name__ == "__main__":
    main()