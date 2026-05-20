import os
import re
import socket
import httpx
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from googleapiclient.discovery import build
from pydantic_settings import BaseSettings, SettingsConfigDict  # ✅ Import Settings

# ────── 1. SETTINGS & CREDENTIALS (Loaded from .env) ──────
class Settings(BaseSettings):
    """Loads environment variables from .env file."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allows .env keys to be case-insensitive
        extra="ignore",
    )

    youtube_api_key: str
    webshare_proxy_username: str
    webshare_proxy_password: str
    rapidapi_key: str

# Initialize settings
settings = Settings()

# ────── 2. GLOBAL PROXY (For Google API client only) ──────
# We use settings variables here
PROXY_URL = f"http://{settings.webshare_proxy_username}:{settings.webshare_proxy_password}@p.webshare.io:80"
os.environ["HTTPS_PROXY"] = PROXY_URL
os.environ["HTTP_PROXY"]  = PROXY_URL

# ────── 3. DNS CHECK ──────
print("[STARTUP] Checking Proxy DNS resolution...")
try:
    socket.gethostbyname("p.webshare.io")
    print("[STARTUP] ✅ Proxy DNS resolved.")
except socket.gaierror:
    print("[STARTUP] ❌ Cannot resolve proxy host.")

app = FastAPI(title="YouTube Data Fetcher")

# ────── 4. MODELS ──────
class VideoEverythingResponse(BaseModel):
    video_id:      str
    title:         str
    description:   str
    hashtags:      str
    tags:          List[str]
    transcript:    Optional[str]
    thumbnail_url: Optional[str]

# ────── 5. FUNCTIONS ──────

def fetch_metadata(video_id: str) -> Optional[dict]:
    """
    Fetches video metadata via Google YouTube Data API v3.
    Routes through Webshare proxy via global env vars (needed on Railway).
    """
    try:
        print(f"[METADATA] Fetching via Google API...")
        # ✅ Use settings key here
        youtube  = build("youtube", "v3", developerKey=settings.youtube_api_key)
        response = youtube.videos().list(part="snippet", id=video_id).execute()

        if not response.get("items"):
            print(f"[METADATA] No items found for video_id={video_id}")
            return None

        item   = response["items"][0]["snippet"]
        thumbs = item.get("thumbnails", {})
        thumbnail_url = (
            thumbs.get("maxres",   {}).get("url") or
            thumbs.get("standard", {}).get("url") or
            thumbs.get("high",     {}).get("url") or
            thumbs.get("default",  {}).get("url")
        )

        print("[METADATA] ✅ Success.")
        return {
            "title":         item.get("title", ""),
            "description":   item.get("description", ""),
            "tags":          item.get("tags", []),
            "thumbnail_url": thumbnail_url,
        }
    except Exception as e:
        print(f"[METADATA][ERROR] {e}")
        return None


def fetch_transcript(video_id: str) -> Optional[str]:
    """
    Fetches transcript via RapidAPI (youtube-transcriptor).

    ── Why trust_env=False ──────────────────────────────────────────────────
    httpx respects HTTP_PROXY/HTTPS_PROXY env vars by default — same as requests.
    Since we set those globally for the Google API client, every httpx call
    would also route through Webshare unless we explicitly opt out.
    RapidAPI is a public HTTPS endpoint; it needs no proxy.
    trust_env=False tells httpx: "ignore all proxy env vars for this client."
    ─────────────────────────────────────────────────────────────────────────
    """
    url     = "https://youtube-transcriptor.p.rapidapi.com/transcript"
    headers = {
        "x-rapidapi-host": "youtube-transcriptor.p.rapidapi.com",
        # ✅ Use settings key here
        "x-rapidapi-key":  settings.rapidapi_key,
    }
    params = {"video_id": video_id, "lang": "en"}

    try:
        print(f"[TRANSCRIPT] Fetching via RapidAPI...")

        # trust_env=False → httpx ignores HTTP_PROXY/HTTPS_PROXY env vars
        with httpx.Client(timeout=30.0, trust_env=False) as client:
            response = client.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

        transcript_text = _parse_transcript_response(data)

        if not transcript_text:
            print("[TRANSCRIPT] ⚠️  Parsed transcript is empty.")
            return None

        print(f"[TRANSCRIPT] ✅ Success — {len(transcript_text)} chars.")
        return transcript_text.strip()

    except httpx.HTTPStatusError as e:
        print(f"[TRANSCRIPT][HTTP ERROR] {e.response.status_code}: {e.response.text[:300]}")
        return None
    except Exception as e:
        print(f"[TRANSCRIPT][ERROR] {e}")
        return None


def _parse_transcript_response(data) -> str:
    """
    Parses the RapidAPI transcript response into a plain string.
    """
    # ── Case 1 & 2: Outer list ──
    if isinstance(data, list) and len(data) > 0:
        first = data[0]
        if isinstance(first, dict) and "transcriptionAsText" in first:
            return first["transcriptionAsText"]
        if isinstance(first, dict) and "transcripts" in first:
            segments = first["transcripts"]
            if isinstance(segments, list):
                return " ".join(s.get("text", "") for s in segments)
        return str(first)

    # ── Case 3: Dict at top level ──
    if isinstance(data, dict):
        if "transcriptionAsText" in data:
            return data["transcriptionAsText"]
        if "transcripts" in data:
            val = data["transcripts"]
            if isinstance(val, list):
                return " ".join(s.get("text", "") for s in val)
            return str(val)
        if "transcript" in data:
            val = data["transcript"]
            if isinstance(val, list):
                return " ".join(s.get("text", "") for s in val)
            return str(val)

    print(f"[TRANSCRIPT][WARN] Unexpected response shape: {type(data)}")
    return str(data)


def generate_hashtags(tags: List[str], description: str) -> str:
    hashtags_from_tags = ["#" + t.replace(" ", "") for t in tags]
    hashtags_from_desc = re.findall(r"#\w+", description)
    all_hashtags = list(set(hashtags_from_tags + hashtags_from_desc))
    return ", ".join(all_hashtags)


# ────── 6. ENDPOINT ──────
@app.get("/everything/{video_id}", response_model=VideoEverythingResponse)
def get_everything_endpoint(video_id: str):
    print(f"\n[REQUEST] Fetching EVERYTHING for: {video_id}")

    metadata = fetch_metadata(video_id)
    if not metadata:
        raise HTTPException(
            status_code=404,
            detail="Video not found or metadata inaccessible."
        )

    transcript = fetch_transcript(video_id)
    hashtags   = generate_hashtags(metadata["tags"], metadata["description"])

    return VideoEverythingResponse(
        video_id      = video_id,
        title         = metadata["title"],
        description   = metadata["description"],
        hashtags      = hashtags,
        tags          = metadata["tags"],
        transcript    = transcript,
        thumbnail_url = metadata["thumbnail_url"],
    )
