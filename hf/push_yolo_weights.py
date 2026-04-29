from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file


MODEL_PATH = Path(
    "/Users/your_user_name/Documents/source/stacks_inventory/runs/obb/"
    "minh_freeze_extended_imagsz_640_seed_42_100_lr_0.001_freeze_11/weights/best.pt"
)
REPO_ID = "quist99/book-spine-detector-yolov8"
PRIVATE = False


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    api = HfApi()
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Not logged in to Hugging Face. Run: hf auth login"
        ) from exc

    create_repo(repo_id=REPO_ID, repo_type="model", private=PRIVATE, exist_ok=True)

    uploaded_path = upload_file(
        path_or_fileobj=str(MODEL_PATH),
        path_in_repo=MODEL_PATH.name,
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"Uploaded: {uploaded_path}")


if __name__ == "__main__":
    main()