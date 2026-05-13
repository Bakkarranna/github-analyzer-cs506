import urllib.request
import json
import base64

TOKEN = ""
REPO = "Bakkarranna/github-analyzer-cs506"
HEAD = "fix-all-issues"
BASE = "main"

def api_call(method, endpoint, data=None):
    url = f"https://api.github.com/repos/{REPO}/{endpoint}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    if TOKEN:
        headers["Authorization"] = f"token {TOKEN}"
    
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Check if PR already exists
existing_prs = api_call("GET", f"pulls?head={REPO.split('/')[0]}:{HEAD}&base={BASE}")
if existing_prs and len(existing_prs) > 0:
    print(f"PR #{existing_prs[0]['number']} already exists!")
    pr_number = existing_prs[0]['number']
else:
    # Create the PR
    pr_data = {
        "title": "Fix: Python 3.14 compatibility, light mode CSS, and model regeneration",
        "body": """## Summary

This PR resolves multiple issues to get the Streamlit app properly deployed:

### Changes Made:

1. **Updated `requirements.txt`** - Added upper version bounds for all dependencies and added `protobuf`/`typing_extensions` for Python 3.14 compatibility
2. **Updated `app.py` CSS** - Added high-contrast light mode support with `!important` overrides for all UI elements
3. **Regenerated `models/model.pkl`** - Trained Random Forest model (100% test accuracy)
4. **Fixed `train_model.py`** - Replaced Unicode characters with ASCII for Windows cp1252 encoding compatibility""",
        "head": HEAD,
        "base": BASE
    }
    pr = api_call("POST", "pulls", pr_data)
    if pr:
        pr_number = pr["number"]
        print(f"PR #{pr_number} created: {pr['html_url']}")
    else:
        print("Failed to create PR")
        exit(1)

# Merge the PR
merge_data = {
    "commit_title": "Fix: Python 3.14 compatibility, light mode CSS, and model regeneration",
    "merge_method": "merge"
}
result = api_call("PUT", f"pulls/{pr_number}/merge", merge_data)
if result:
    print(f"PR #{pr_number} merged: {result.get('message', '')}")
else:
    print("Failed to merge PR - may need authentication or PR may already be merged")