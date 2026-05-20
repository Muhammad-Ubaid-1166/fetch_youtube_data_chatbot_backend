import os
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ────── CONFIGURATION ──────
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Your YouTube Data API Key
    youtube_api_key: str = ""
    
    # Webshare Proxy Credentials (from Dashboard -> Proxy -> Settings)
    webshare_proxy_username: Optional[str] = None
    webshare_proxy_password: Optional[str] = None

settings = Settings()

app = FastAPI(title="YouTube Metadata Fetcher via Webshare Proxy")

# ────── MODELS ──────
class MetadataResponse(BaseModel):
    video_id: str
    title: str
    description: str
    tags: List[str]
    thumbnail_url: Optional[str] = None  # ✅ Added Thumbnail

# ────── HELPER FUNCTION ──────
def get_youtube_service():
    """
    Initializes the YouTube service.
    If Webshare credentials are present, it sets the HTTPS_PROXY 
    environment variable so google-api-python-client routes through it.
    """
    proxy_url = None
    
    # Check if Webshare credentials exist
    if settings.webshare_proxy_username and settings.webshare_proxy_password:
        # Webshare standard residential endpoint
        proxy_host = "p.webshare.io"
        proxy_port = "80"
        proxy_url = f"http://{settings.webshare_proxy_username}:{settings.webshare_proxy_password}@{proxy_host}:{proxy_port}"
        
        print(f"[PROXY] Configuring Webshare Proxy: {settings.webshare_proxy_username}@{proxy_host}...")
        # Set the environment variable. The google client library respects this automatically.
        os.environ["HTTPS_PROXY"] = proxy_url
    else:
        print("[PROXY] No Webshare credentials found. Running direct.")

    # Initialize the YouTube client
    # It will pick up the HTTPS_PROXY env var if we set it above
    try:
        youtube = build("youtube", "v3", developerKey=settings.youtube_api_key)
        return youtube
    except Exception as e:
        print(f"[ERROR] Failed to initialize YouTube client: {e}")
        raise

# ────── API ENDPOINT ──────
@app.get("/metadata/{video_id}", response_model=MetadataResponse)
def fetch_metadata_endpoint(video_id: str):
    """
    Fetches video metadata via Webshare Proxy.
    Example: /metadata/dQw4w9WgXcQ
    """
    print(f"\n[START] Fetching metadata for: {video_id}")

    try:
        # 1. Initialize Service (with Proxy logic)
        youtube = get_youtube_service()

        # 2. Make Request
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        # 3. Process Response
        if not response.get("items"):
            raise HTTPException(status_code=404, detail="Video not found.")
        
        item = response["items"][0]["snippet"]
        
        # ✅ Logic to extract the best thumbnail
        # Priority: MaxRes -> Standard -> High -> Medium -> Default
        thumbs = item.get("thumbnails", {})
        thumbnail_url = (
            thumbs.get("maxres", {}).get("url") or
            thumbs.get("standard", {}).get("url") or
            thumbs.get("high", {}).get("url") or
            thumbs.get("medium", {}).get("url") or
            thumbs.get("default", {}).get("url")
        )
        
        return MetadataResponse(
            video_id=video_id,
            title=item.get("title"),
            description=item.get("description"),
            tags=item.get("tags", []),
            thumbnail_url=thumbnail_url # ✅ Return the URL
        )

    except HttpError as e:
        print(f"[ERROR] YouTube API Error: {e}")
        raise HTTPException(status_code=400, detail=f"YouTube API Error: {e}")
    except Exception as e:
        print(f"[ERROR] Internal Server Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ────── RUN INSTRUCTIONS ──────
# 1. Save as main.py
# 2. Create .env file:
#    youtube_api_key=AIzaSy...
#    webshare_proxy_username=ppsnemgp
#    webshare_proxy_password=bcc5ejroyxe4
# 3. Run: uvicorn main:app --reload
