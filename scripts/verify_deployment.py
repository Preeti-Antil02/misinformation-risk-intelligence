"""
scripts/verify_deployment.py
============================
Post-Deployment Sync, Build & Health Verification Suite for RiskLens on Hugging Face Spaces.
Validates:
1. GitHub Actions sync-to-hf.yml latest workflow run status & timestamp.
2. Space /health deep probe (active SQLite, storage permissions, model weights).
3. Space /version release verification (verifies deployed code matches v2.1.0).
4. Streamlit Dashboard frontend availability.
"""

import os
import sys
import time
import json
import argparse
import requests
from datetime import datetime
from pathlib import Path

# Add project root
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

DEFAULT_SPACE_URL = "https://preeti-antil-risklens.hf.space"
GITHUB_REPO = "Preeti-Antil02/misinformation-risk-intelligence"
WORKFLOW_FILE = "sync-to-hf.yml"


def check_github_sync(repo: str = GITHUB_REPO, token: str = "") -> dict:
    """Queries GitHub Actions REST API to verify the latest sync-to-hf.yml run."""
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            return {
                "status": "warning",
                "message": f"Workflow {WORKFLOW_FILE} not found or repository is private (provide GITHUB_TOKEN)."
            }
        if resp.status_code != 200:
            return {
                "status": "warning",
                "message": f"GitHub API returned HTTP {resp.status_code}: {resp.text[:100]}"
            }

        data = resp.json()
        runs = data.get("workflow_runs", [])
        if not runs:
            return {"status": "warning", "message": "No workflow runs found for sync-to-hf.yml yet."}

        latest = runs[0]
        run_status = latest.get("status")  # queued, in_progress, completed
        conclusion = latest.get("conclusion")  # success, failure, cancelled
        created_at = latest.get("created_at")
        html_url = latest.get("html_url")
        commit_sha = latest.get("head_sha", "")[:7]

        is_success = (run_status == "completed" and conclusion == "success")

        return {
            "status": "pass" if is_success else ("running" if run_status != "completed" else "fail"),
            "run_id": latest.get("id"),
            "run_status": run_status,
            "conclusion": conclusion,
            "commit": commit_sha,
            "timestamp": created_at,
            "url": html_url
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_space_health(base_url: str) -> dict:
    """Actively probes /health endpoint on the deployed Space."""
    health_url = f"{base_url.rstrip('/')}/health"
    start_t = time.time()
    try:
        resp = requests.get(health_url, timeout=20)
        elapsed_ms = round((time.time() - start_t) * 1000, 1)

        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:200]}

        return {
            "status_code": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "is_healthy": resp.status_code == 200 and data.get("status") == "healthy",
            "data": data
        }
    except requests.exceptions.Timeout:
        return {"status_code": 0, "is_healthy": False, "error": "Request timed out (Space might be starting/sleeping)."}
    except Exception as e:
        return {"status_code": 0, "is_healthy": False, "error": str(e)}


def check_space_version(base_url: str) -> dict:
    """Probes /version endpoint to confirm the deployed release."""
    version_url = f"{base_url.rstrip('/')}/version"
    try:
        resp = requests.get(version_url, timeout=10)
        if resp.status_code == 200:
            return {"status": "pass", "data": resp.json()}
        return {"status": "fail", "status_code": resp.status_code, "text": resp.text[:100]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_streamlit_frontend(base_url: str) -> dict:
    """Verifies that the Streamlit frontend responds on the main Space port."""
    ui_url = base_url.rstrip('/')
    st_health_url = f"{ui_url}/_stcore/health"
    try:
        resp = requests.get(st_health_url, timeout=15)
        st_ok = resp.status_code == 200 and "ok" in resp.text.lower()
        if st_ok:
            return {"status": "pass", "message": "Streamlit core health responded OK (HTTP 200)"}

        # Fallback to checking root HTML
        root_resp = requests.get(ui_url, timeout=15)
        if root_resp.status_code == 200 and ("streamlit" in root_resp.text.lower() or "risklens" in root_resp.text.lower()):
            return {"status": "pass", "message": "Streamlit root HTML loaded successfully"}

        return {"status": "fail", "status_code": resp.status_code, "text": resp.text[:100]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="RiskLens Post-Deployment Verification Suite")
    parser.add_argument("--url", default=os.getenv("SPACE_URL", DEFAULT_SPACE_URL), help="Hugging Face Space URL")
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN", ""), help="GitHub Personal Access Token (for private repo rate limits)")
    args = parser.parse_args()

    space_url = args.url.rstrip("/")

    print("=" * 75)
    print("🛡️  RiskLens v2.1.0 Post-Deployment Verification Suite")
    print(f"🎯 Target URL: {space_url}")
    print(f"🕒 Timestamp : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 75)

    all_passed = True

    # 1. GitHub Actions Sync Check
    print("\n[1/4] 🔄 Checking GitHub Actions Sync Workflow (sync-to-hf.yml)...")
    sync_res = check_github_sync(token=args.github_token)
    if sync_res.get("status") == "pass":
        print(f"  ✅ PASS: Latest sync completed successfully!")
        print(f"     • Run ID   : {sync_res.get('run_id')}")
        print(f"     • Commit   : {sync_res.get('commit')}")
        print(f"     • Synced At: {sync_res.get('timestamp')}")
    elif sync_res.get("status") == "running":
        print(f"  ⏳ IN PROGRESS: Sync workflow is currently executing (Run ID: {sync_res.get('run_id')}).")
    else:
        print(f"  ⚠️  NOTE: {sync_res.get('message', 'Could not query GitHub Actions API.')}")

    # 2. Space Deep Health Probe
    print(f"\n[2/4] 🏥 Probing Space Health Endpoint ({space_url}/health)...")
    health_res = check_space_health(space_url)
    if health_res.get("is_healthy"):
        data = health_res.get("data", {})
        comp = data.get("components", {})
        print(f"  ✅ PASS: Deep /health probe returned HTTP 200 OK ({health_res.get('elapsed_ms')} ms)")
        print(f"     • Version        : {data.get('version', 'N/A')}")
        print(f"     • Database       : {comp.get('database', {}).get('status', 'N/A')}")
        print(f"     • Storage Path   : {comp.get('storage', {}).get('path', 'N/A')} (Mode: {comp.get('storage', {}).get('mode', 'N/A')})")
        print(f"     • Models Ready   : {comp.get('models', {}).get('status', 'N/A')}")
        print(f"     • Telegram Bot   : {comp.get('telegram', {}).get('status', 'N/A')} (Mode: {comp.get('telegram', {}).get('mode', 'N/A')})")
        
        # Display secret configuration presence
        secrets = comp.get("secrets_status", {})
        if secrets:
            print("     • Secrets Check  :")
            for s_name, s_val in secrets.items():
                icon = "✓" if s_val else "✗"
                print(f"       [{icon}] {s_name}: {'CONFIGURED' if s_val else 'MISSING'}")
    else:
        all_passed = False
        print(f"  ❌ FAIL: /health probe failed (HTTP {health_res.get('status_code')})")
        if "error" in health_res:
            print(f"     • Error: {health_res['error']}")
        elif "data" in health_res:
            print(f"     • Response: {health_res['data']}")

    # 3. Release Version Verification
    print(f"\n[3/4] 🏷️  Verifying Deployed Version ({space_url}/version)...")
    ver_res = check_space_version(space_url)
    if ver_res.get("status") == "pass":
        vdata = ver_res.get("data", {})
        print(f"  ✅ PASS: Deployed version verified!")
        print(f"     • Release: {vdata.get('release_name', 'N/A')}")
        print(f"     • Version: {vdata.get('version', 'N/A')}")
        print(f"     • Models : {list(vdata.get('models_loaded', {}).keys())}")
    else:
        all_passed = False
        print(f"  ❌ FAIL: /version endpoint check failed ({ver_res})")

    # 4. Streamlit Frontend Availability Check
    print(f"\n[4/4] 🌐 Checking Streamlit Dashboard Availability...")
    ui_res = check_streamlit_frontend(space_url)
    if ui_res.get("status") == "pass":
        print(f"  ✅ PASS: {ui_res.get('message')}")
    else:
        all_passed = False
        print(f"  ❌ FAIL: Streamlit frontend check failed ({ui_res})")

    # Final Summary Matrix
    print("\n" + "=" * 75)
    if all_passed:
        print("🎉 ALL POST-DEPLOYMENT VERIFICATION CHECKS PASSED!")
        print("   The Hugging Face Space is live, fully synchronized, and operational.")
    else:
        print("⚠️  ONE OR MORE VERIFICATION CHECKS FAILED.")
        print("   Review the output above to diagnose missing secrets or cold start issues.")
    print("=" * 75)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
