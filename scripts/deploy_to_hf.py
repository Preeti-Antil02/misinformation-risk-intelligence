"""
scripts/deploy_to_hf.py
=======================
Creates a clean, single-commit orphan deployment branch containing ONLY the
current working directory (<4 MB) and force pushes it to Hugging Face Spaces.
Guarantees 0 MB legacy LFS objects or history bloat.
"""

import sys
import subprocess
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token_path.exists():
        print(f"Error: Token not found at {token_path}")
        sys.exit(1)
        
    token = token_path.read_text().strip()
    hf_remote_url = f"https://Preeti-Antil:{token}@huggingface.co/spaces/Preeti-Antil/RiskLens"

    print("[1/4] Creating clean orphan deployment branch 'hf-deploy'...")
    subprocess.run(["git", "branch", "-D", "hf-deploy"], cwd=BASE_DIR, capture_output=True)
    subprocess.run(["git", "checkout", "--orphan", "hf-deploy"], cwd=BASE_DIR, check=True, capture_output=True)
    
    print("[2/4] Staging clean codebase (<4MB total)...")
    subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True)
    
    print("[3/4] Committing single deployment snapshot...")
    commit_res = subprocess.run(
        ["git", "commit", "-m", "deploy: RiskLens v2.1.0 Enterprise Intelligence Multi-Container"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    print(commit_res.stdout)
    
    print("[4/4] Force pushing clean tree to Hugging Face Space (Preeti-Antil/RiskLens)...")
    push_res = subprocess.run(
        ["git", "push", hf_remote_url, "hf-deploy:main", "--force"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    
    # Return back to main branch
    subprocess.run(["git", "checkout", "main"], cwd=BASE_DIR, capture_output=True)
    subprocess.run(["git", "branch", "-D", "hf-deploy"], cwd=BASE_DIR, capture_output=True)
    
    if push_res.returncode == 0:
        print("✅ SUCCESS: Deployed to Hugging Face Space!")
        print(push_res.stdout)
        print(push_res.stderr)
    else:
        print("❌ Push error:")
        print(push_res.stdout)
        print(push_res.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
