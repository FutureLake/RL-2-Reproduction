"""Fetch the unmodified reference checkout; never vendor it into this repository."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "third_party/pytorch_rl2"
SHA = "51afa4b6c6d05b469c7f74076b9490ae0809b289"

if __name__ == "__main__":
    if TARGET.exists():
        raise FileExistsError(f"Inspect the existing checkout first: {TARGET}")
    TARGET.parent.mkdir(exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/lucaslingle/pytorch_rl2.git", str(TARGET)], check=True)
    subprocess.run(["git", "-C", str(TARGET), "checkout", "--detach", SHA], check=True)
    print("Reference checkout ready:", SHA)
