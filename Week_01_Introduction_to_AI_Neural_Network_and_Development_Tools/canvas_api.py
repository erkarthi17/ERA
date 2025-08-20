import os
import requests
from typing import Dict, Any, List
from pathlib import Path
from summarizer import summarize_text
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    def load_dotenv(*args, **kwargs):
        return False


def _get_config(overrides: Dict[str, str] | None = None):
    # Ensure .env is loaded for each call (robust on Windows editors/encodings)
    try:
        env_path = str(Path(__file__).resolve().parent / ".env")
        load_dotenv(dotenv_path=env_path, override=True)
    except Exception:
        pass

    env = {
        "CANVAS_API_URL": os.environ.get("CANVAS_API_URL", "https://canvas.instructure.com"),
        "CANVAS_API_TOKEN": os.environ.get("CANVAS_API_TOKEN", ""),
        "CANVAS_COURSE_ID": os.environ.get("CANVAS_COURSE_ID", ""),
    }
    # Ignore overrides to enforce env-based config
    merged = env

    base_url = (merged.get("CANVAS_API_URL") or "").strip().strip('"').strip("'")
    access_token = (merged.get("CANVAS_API_TOKEN") or "").strip().strip('"').strip("'")
    course_id = (merged.get("CANVAS_COURSE_ID") or "").strip().strip('"').strip("'")
    api_url = f"{base_url.rstrip('/')}/api/v1"
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    return api_url, headers, access_token, course_id


def _get_json(url: str, headers: Dict[str, str], params: Dict[str, Any] | None = None) -> Any:
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _list_pages(api_url: str, headers: Dict[str, str], course_id: str) -> List[Dict[str, Any]]:
    url = f"{api_url}/courses/{course_id}/pages/"
    # Request many items per page to reduce calls; add simple pagination if needed later
    params = {"per_page": 100}
    return _get_json(url, headers=headers, params=params)


def _get_page(api_url: str, headers: Dict[str, str], course_id: str, page_url: str) -> Dict[str, Any]:
    url = f"{api_url}/courses/{course_id}/pages/{page_url}"
    return _get_json(url, headers=headers)


def _map_page_to_module(api_url: str, headers: Dict[str, str], course_id: str) -> Dict[str, str]:
    """Return mapping from page_url -> module (session) title."""
    mapping: Dict[str, str] = {}
    modules_url = f"{api_url}/courses/{course_id}/modules"
    modules = _get_json(modules_url, headers=headers, params={"per_page": 100})
    if not isinstance(modules, list):
        return mapping
    for module in modules:
        module_id = module.get("id")
        module_name = module.get("name") or "Session"
        if not module_id:
            continue
        items_url = f"{api_url}/courses/{course_id}/modules/{module_id}/items"
        items = _get_json(items_url, headers=headers, params={"per_page": 100})
        if not isinstance(items, list):
            continue
        for item in items:
            if item.get("type") == "Page" and item.get("page_url"):
                mapping[item["page_url"]] = module_name
    return mapping
    
def search_canvas(keyword: str, overrides: Dict[str, str] | None = None) -> Dict[str, str]:
    api_url, headers, access_token, course_id = _get_config(None)
    if not access_token or not course_id:
        return {
            "title": "Configuration error",
            "content": (
                "Missing CANVAS_ACCESS_TOKEN or CANVAS_COURSE_ID. "
                "Set them as environment variables before starting the server."
            ),
        }

    keyword_lower = (keyword or "").strip().lower()
    if not keyword_lower:
        return {"title": "Invalid input", "content": "Keyword must not be empty."}

    try:
        pages = _list_pages(api_url, headers, course_id)
        if not isinstance(pages, list) or not pages:
            return {"title": "Not Found", "content": "No pages found or access denied."}

        page_to_module = _map_page_to_module(api_url, headers, course_id)

        for page in pages:
            page_url = page.get("url")
            if not page_url:
                continue
            page_data = _get_page(api_url, headers, course_id, page_url)
            body = page_data.get("body", "")
            title = page_data.get("title", "Untitled")

            # Find keyword in body and extract snippet
            body_lower = body.lower()
            idx = body_lower.find(keyword_lower)
            if idx != -1 or keyword_lower in title.lower():
                session_title = page_to_module.get(page_url) or title

                # If keyword found in body, extract snippet
                if idx != -1:
                    start = max(0, idx - 100)
                    end = min(len(body), idx + 100)
                    snippet = body[start:end]
                else:
                    snippet = body

                summary = summarize_text(snippet)
                print("Summary found out is :", summary)
                print("Session Title is :", session_title)
                print("Title is :", title)
                return {
                    "title": title,
                    "content": body,
                    "session": session_title,
                    "snippet": snippet,
                    "summary": summary,
                }

        return {"title": "Not Found", "content": "No matching keyword found in any page."}
    except requests.HTTPError as http_err:
        return {"title": "HTTP error", "content": str(http_err)}
    except Exception as exc:
        return {"title": "Error", "content": str(exc)}