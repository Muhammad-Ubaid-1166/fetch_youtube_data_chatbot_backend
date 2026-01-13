import re
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Tuple
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

# --- NEW IMPORTS for Image Analysis ---
from urllib.parse import urlparse, parse_qs
import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# --- NEW IMPORT for OpenAI ---
from openai import OpenAI

# --- REAL AGENT IMPORTS ---
from agents import Runner, Agent
from config import first_model,second_model,third_model,fourth_model,fifth_model


# --- Configuration ---
YOUTUBE_API_KEY = "AIzaSyDIkbicF6fP0kazXz__FTDeEY-5nFAiy_A"
# GEMINI_API_KEY = "AIzaSyBR40AVO4lVWCuHMqaLMpJZd_UgE0jSgd4"
GENAI_MODEL_NAME = "gemini-2.5-flash"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# --- Pydantic Models for Request and Response ---

# --- UPDATED REQUEST MODEL ---
# Added fields for user-defined instructions.
class YouTubeURLRequest(BaseModel):
    url: str
    language: str = "English"
    metadata_instruction: str = "Rewrite and improve the title, description, and hashtags. The description should be about 300 words. Provide exactly 4 relevant hashtags."
    transcript_instruction: str = "Rewrite the transcript into clear, formal, and professional language. Preserve the original meaning and flow."

class ChangerOutputType(BaseModel):
    title: str = Field(description="A short, catchy title for the content.")
    description: str = Field(description="A detailed description of the content which must based on 300 words.")
    hashtags: List[str] = Field(description="A list of relevant 4 hashtags. , hashtags must be 4")

class TranscriptOutputType(BaseModel):
    transcription: str

# --- NEW MODELS FOR IMAGE DESCRIPTIONS ---
class ImageItem(BaseModel):
    title: str
    description: str

class ImagesOutputTypes(BaseModel):
    images: List[ImageItem]

# --- SIMPLIFIED RESPONSE MODEL ---
# Removed fields related to the old translation features.
class CombinedOutputType(BaseModel):
    metadata: ChangerOutputType = Field(description="Processed title, description, and hashtags.")
    transcript: TranscriptOutputType = Field(description="The rewritten full transcript.")
    image_description: Optional[str] = Field(default=None, description="A detailed description of the video's thumbnail image.")
    generated_image_url: Optional[str] = Field(default=None, description="The URL of the AI-generated image based on the thumbnail description.")
    # --- NEW FIELD FOR IMAGE DESCRIPTIONS ---
    image_descriptions: Optional[ImagesOutputTypes] = Field(default=None, description="List of image descriptions for highlighted transcript sections.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="YouTube Content Processor (with Custom Instructions)",
    description="Fetches metadata and transcript, then uses LLM agents with user-defined instructions to process them.",
    version="11.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# --- Helper Functions (Unchanged) ---
def extract_video_id(url: str) -> Optional[str]:
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_metadata(video_id: str) -> Optional[dict]:
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if not response.get("items"): return None
        item = response["items"][0]["snippet"]
        return {"title": item["title"], "description": item["description"], "tags": item.get("tags", [])}
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return None

def build_metadata_string(metadata: dict) -> str:
    title = metadata["title"]; description = metadata["description"]; tags = metadata.get("tags", [])
    hashtags_from_desc = re.findall(r"#\w+", description)
    hashtags_from_tags = ["#" + t.replace(" ", "") for t in tags]
    all_hashtags = ', '.join(list(set(hashtags_from_desc + hashtags_from_tags)))
    return f"Title: {title}\n\nDescription: {description}\nAll Hashtags: {all_hashtags}"

import requests

def fetch_transcript(video_id: str) -> Optional[str]:
   
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id, languages=['en', 'en-GB'])
    full_text = " ".join([snippet.text for snippet in transcript_list])
    return full_text.strip()
  
def fetch_thumbnail_url(url: str) -> Optional[str]:
    try:
        parsed_url = urlparse(url)
        video_id = None
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path.lstrip('/')
        
        if video_id:
            return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        return None
    except Exception as e:
        print(f"Error parsing URL for thumbnail: {e}")
        return None

import requests
from PIL import Image
from io import BytesIO
import google.generativeai as genai
from agents import function_tool, Agent
# Configure Gemini
genai.configure(api_key="AIzaSyA2An2o9RC9K3ilYY_I3X1F7nhDE9LCW-E")
MODEL_NAME = "gemini-2.5-flash"


@function_tool
def image_recognition_tool(image_url: str) -> str:
    """
    Uses Gemini Vision to analyze the image and return
    a human-like visual description.
    """
    try:
        # Load image (PIL is ONLY a loader here)
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Gemini Vision model
        model = genai.GenerativeModel(MODEL_NAME)

        prompt = (
            "Describe exactly what you see in this image. "
            "Mention people, objects, and their positions "
            "(left, right, center). Be clear and specific."
        )

        result = model.generate_content([prompt, image])

        return result.text.strip()

    except Exception as e:
        return f"Image analysis failed: {str(e)}"


image_analyzer_agent = Agent(
    name="Image Visual Analyzer",
    instructions=(
        "You are a vision-based image analyzer.\n"
        "You MUST use the image_recognition_tool.\n\n"
        "Your job is to return a clear, human-readable "
        "description of what is visible in the image.\n\n"
        "Rules:\n"
        "- Describe people and objects naturally\n"
        "- Mention left/right/center placement\n"
        "- Do NOT talk about pixels, brightness, or analysis\n"
        "- Output 1–3 clear sentences only\n\n"
        "Example:\n"
        "'The image shows a girl standing near the center of the room, "
        "with a bed positioned on the right side.'"
        "output should like title , description which must be detail description which contain 80 words"
    ),
    model=first_model,
    tools=[image_recognition_tool]
)

import asyncio
from agents import Runner

def changeragent(image_url):
  

    result = Runner.run_sync(
        image_analyzer_agent,
        f"Analyze this image: {image_url}"
    )

    return result.final_output




def generate_image_with_openai(prompt: str) -> Optional[str]:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        MODEL = "dall-e-3"
        SIZE =  "1024x1024"
        QUALITY = "standard"
        print(f"Generating image for prompt: '{prompt}")
        response = client.images.generate(model=MODEL, prompt=prompt, size=SIZE, quality=QUALITY, n=1)
        image_url = response.data[0].url
        print(f"Successfully generated image. URL: {image_url}")
        return image_url
    except Exception as e:
        print(f"An error occurred during image generation with OpenAI: {e}")
        return None

# --- UPDATED: Dynamic Agent Creation Function ---
def create_agents(metadata_instruction: str, transcript_instruction: str, language: str) -> Tuple[Agent, Agent, Agent, Agent]:
    """Creates and returns agents configured with user-defined instructions and a target language."""
    
    # Provide robust defaults in case the user provides an empty instruction
    default_metadata_instruction = ""
    default_transcript_instruction = ""

    final_metadata_instruction = metadata_instruction if metadata_instruction.strip() else default_metadata_instruction
    final_transcript_instruction = transcript_instruction if transcript_instruction.strip() else default_transcript_instruction

    # Make the language requirement more prominent and explicit
    language_requirement = f"IMPORTANT: The entire output MUST be in {language}. Do not use any other language. "
    
    # Prepend the language requirement to make it more prominent
    final_metadata_instruction = language_requirement + final_metadata_instruction
    final_transcript_instruction = language_requirement + final_transcript_instruction

    title_des_hashtag_changer = Agent(
        name="Title, Description, and Hashtag Modifier",
        instructions=final_metadata_instruction,
        output_type=ChangerOutputType,
        model=second_model
    )

    transcript_rewrite_agent = Agent(
        name="Transcript Rewriter",
        instructions=final_transcript_instruction,
        output_type=TranscriptOutputType,
        model=third_model
    )
    
    # FIXED: Changed the transcribe_image_higliter_agent to not use output_type
    transcribe_image_higliter_agent = Agent(
        name="transcribe image highlighter",
        instructions="""
    You are an expert video editing assistant.

    TASK:
    - Carefully read the provided transcript.
    - Identify the most natural and meaningful moments where an image should appear on screen.

    FORMATTING RULES:
    - Do NOT place two highlighted image markers back-to-back; merge them into one if they represent the same idea.
    - Highlight ONLY the exact sentence or phrase where the image should appear.
    - Wrap the highlighted text using **{BOLD CURLY BRACKETS}**.
    - Do NOT rewrite, summarize, or rephrase the transcript.
    - Do NOT add explanations or comments.
    - Preserve the transcript word-for-word.

    LIMITS:
    - Highlight a maximum of 12 image placements.

    OUTPUT:
    - Return the full transcript with highlighted image placements only.
    """,
        model=fourth_model  # Removed output_type specification
    )
    
    # --- NEW AGENT FOR IMAGE DESCRIPTIONS ---
    generate_image_description = Agent(
        name="generate image description",
        instructions=f"""
    You will receive a transcript containing image markers wrapped in {{curly braces}}.

    TASK:
    - Each {{curly-braced}} section represents ONE image.
    - For EACH image marker:
      • Generate a short, clear image title
      • Generate a concise image description suitable for image generation

    RULES:
    - The number of titles and descriptions MUST exactly match the number of image markers.
    - Do NOT include anything except the title and description lists.
    - {language_requirement}
    """,
        output_type=ImagesOutputTypes,
        model=fifth_model
    )
    
    return title_des_hashtag_changer, transcript_rewrite_agent, transcribe_image_higliter_agent, generate_image_description

# --- Main API Endpoint (UPDATED) ---
@app.post("/process-video-completely", response_model=CombinedOutputType)
def process_video_completely(request: YouTubeURLRequest):
    try:
        video_id = extract_video_id(request.url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL. Could not extract video ID.")

        # --- Create agents with user-defined instructions ---
        title_des_hashtag_changer, transcript_rewrite_agent, transcribe_image_higliter_agent, generate_image_description = create_agents(
            request.metadata_instruction,
            request.transcript_instruction,
            request.language
        )
        print(f"Processing video with custom instructions in language: {request.language}")

        # --- Path 1: Process Metadata ---
        metadata = fetch_metadata(video_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Video not found or metadata is inaccessible.")
        metadata_string = build_metadata_string(metadata)
        processed_metadata = Runner.run_sync(title_des_hashtag_changer, f"here is input ({metadata_string})").final_output

        # --- Path 2: Process Transcript ---
        raw_transcript = fetch_transcript(video_id)
        if not raw_transcript:
            raise HTTPException(status_code=404, detail="Could not find a transcript for this video.")
        
        # Get the rewritten transcript
        rewritten_transcript = Runner.run_sync(transcript_rewrite_agent, f"Here is the transcript , rewrite it in your words , and it must be based 2,500 words:\n\n{raw_transcript}").final_output
        
        # FIXED: Get the highlighted transcript as a string and create TranscriptOutputType manually
        highlighted_transcript = Runner.run_sync(transcribe_image_higliter_agent, rewritten_transcript.transcription).final_output
        final_transcript = TranscriptOutputType(transcription=highlighted_transcript)
        
        # --- NEW: Generate Image Descriptions ---
        image_descriptions = None
        if highlighted_transcript:
            image_descriptions = Runner.run_sync(generate_image_description, highlighted_transcript).final_output
        
        # --- Path 3: Process Thumbnail Image & Generate New Image ---
        thumbnail_url = fetch_thumbnail_url(request.url)
        image_description = None
        generated_image_url = None
        
        if thumbnail_url:
            image_description = changeragent(thumbnail_url)
            if image_description:
                dalle_prompt = image_description
                generated_image_url = generate_image_with_openai(dalle_prompt)
        
        
        return CombinedOutputType(
            metadata=processed_metadata, 
            transcript=final_transcript,
            image_description=image_description,
            generated_image_url=generated_image_url,
            image_descriptions=image_descriptions  # Add the new field
        )

    except HttpError as e: 
        raise HTTPException(status_code=503, detail=f"YouTube API error: {e}")
    except ValidationError as e:
        print(f"Pydantic Validation Error: {e.json()}")
        raise HTTPException(status_code=500, detail=f"Response validation failed: {e}")
    except Exception as e:
        print(f"An unexpected internal error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# Added a health check endpoint to verify the server is running
@app.get("/health")
def health_check():
    return {"status": "healthy"}