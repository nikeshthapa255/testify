import shutil
from pathlib import Path

def clear_huggingface_cache():
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print(f"Clearing Hugging Face cache at {cache_dir}...")
        shutil.rmtree(cache_dir)
        print("Cache cleared.")
    else:
        print("No cache directory found.")

if __name__ == "__main__":
    clear_huggingface_cache()
