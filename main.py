"""
YouTube Data Fetcher + Rewriter — FastAPI Backend (Monolithic)
==============================================================
Professional multi-node LangGraph pipeline that:
  1. Fetches YouTube video metadata, transcript, and thumbnail
  2. Rewrites title, description, and hashtags (3-4% variation) in any language
  3. Plans structured script steps from transcript
  4. Writes a 2500+ word YouTube script sequentially in any language
  5. Polishes the script into one cohesive piece
  6. Plans and generates AI images for each script section
  7. Analyzes thumbnail via Groq Vision and regenerates it
  8. Returns everything in a single API response

All generated content (title, description, hashtags, script) is produced
in the language specified by the client via the `language` parameter.

Endpoint Parameters:
  - url:                  YouTube video URL
  - language:             Target language (e.g., "English", "Urdu", "Hindi", "Arabic")
  - min_script_word_count: Minimum word count target for the script (default: 2500)
  - default_image_count:  Target number of images to generate (default: 15)

Run: uvicorn main:app --reload
API Docs: http://127.0.0.1:8000/docs
"""

import os
import re
import json
import time
import traceback
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from typing import Optional, List, Dict, TypedDict

import httpx
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi
from groq import Groq

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Script Config (defaults — can be overridden per-request) ──
    min_script_word_count: int = 2500
    default_image_count: int = 15

    # ── YouTube Data API ──
    youtube_api_key: str = ""

    # ── Groq API ──
    groq_api_key_1: str = ""
    groq_api_key_2: str = ""
    groq_api_key_3: str = ""

    # ── Google Gemini API ──
    gemini_api_key_1: str = ""
    gemini_api_key_2: str = ""
    genai_model_name: str = "gemini-2.5-flash"

    # ── Cloudflare Workers AI ──
    cf_worker_url: str = ""
    cf_worker_api_key: str = ""
    cf_model: str = "@cf/black-forest-labs/flux-2-max"

    # ── ImgBB ──
    imgbb_api_key: str = ""

    # ── Image Generation Timing ──
    image_gen_delay_sec: int = 1

    # ── LLM Model Names ──
    groq_model_name: str = "llama-3.1-8b-instant"
    groq_vision_model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # ── Gemini OpenAI-compatible Base URL ──
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"


settings = Settings()


def validate_api_keys() -> None:
    """Validate that required API keys are present.
    Raises ValueError if any required key is missing.
    """
    required_keys = {
        "YOUTUBE_API_KEY": settings.youtube_api_key,
        "GROQ_API_KEY_1": settings.groq_api_key_1,
        "GROQ_API_KEY_2": settings.groq_api_key_2,
        "GROQ_API_KEY_3": settings.groq_api_key_3,
        "GEMINI_API_KEY_1": settings.gemini_api_key_1,
        "GEMINI_API_KEY_2": settings.gemini_api_key_2,
        "CF_WORKER_URL": settings.cf_worker_url,
        "IMGBB_API_KEY": settings.imgbb_api_key,
        "CF_WORKER_API_KEY": settings.cf_worker_api_key,
        "OPENAI_API_KEY": settings.openai_api_key,
    }

    missing = [name for name, value in required_keys.items() if not value]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Log loaded keys (masked)
    print(f"[CONFIG][OK] YOUTUBE_API_KEY: {settings.youtube_api_key[:6]}...")
    print(f"[CONFIG][OK] GROQ_API_KEY_1: {settings.groq_api_key_1[:6]}...")
    print(f"[CONFIG][OK] GROQ_API_KEY_2: {settings.groq_api_key_2[:6]}...")
    print(f"[CONFIG][OK] GROQ_API_KEY_3: {settings.groq_api_key_3[:6]}...")
    print(f"[CONFIG][OK] GEMINI_API_KEY_1: {settings.gemini_api_key_1[:6]}...")
    print(f"[CONFIG][OK] GEMINI_API_KEY_2: {settings.gemini_api_key_2[:6]}...")


validate_api_keys()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PYDANTIC SCHEMAS (Request/Response & Structured LLM Outputs)
# ═══════════════════════════════════════════════════════════════════════════════

class YouTubeURLRequest(BaseModel):
    """Incoming request: YouTube URL + pipeline configuration."""
    url: str
    language: str = Field(
        default="English",
        description="Target language for all generated content (title, description, hashtags, script). "
                    "Examples: 'English', 'Urdu', 'Hindi', 'Arabic', 'Spanish', 'French', etc."
    )
    min_script_word_count: int = Field(
        default=2500,
        description="Minimum word count target for the generated script"
    )
    default_image_count: int = Field(
        default=15,
        description="Default target number of images to generate for the script"
    )


class Step(BaseModel):
    """A single step in the script plan."""
    step_number: int = Field(..., description="Sequential step number (1, 2, 3...)")
    description: str = Field(..., description="What this step covers from the transcript")
    continuity_note: str = Field(
        ...,
        description="How this step's output must connect to the previous step's output. "
                    "For Step 1, write 'N/A — this is the opening step.'"
    )
    tone: str = Field(
        ...,
        description="Emotional/delivery tone for this step "
                    "(e.g., energetic, calm, dramatic, humorous, authoritative)"
    )
    word_count: int = Field(..., description="Target word count for this step's script segment")


class TranscriptStepsOutput(BaseModel):
    """Structured output from the transcript steps maker node."""
    video_title: Optional[str] = Field(None, description="Title of the source video")
    total_steps: int = Field(..., description="Total number of steps")
    total_word_count: int = Field(..., description="Sum of all step word counts")
    steps: List[Step] = Field(..., description="Ordered list of script steps")


class ImagePlacement(BaseModel):
    """A single image placed within the transcript."""
    image_number: int = Field(
        ...,
        description="Sequential image number across the ENTIRE transcript (1, 2, 3, ...)"
    )
    step_number: int = Field(
        ...,
        description="Which step this image belongs to"
    )
    placement_context: str = Field(
        ...,
        description="The exact [image-N = visual description] marker as it appears "
                    "in the annotated transcript, e.g. '[image-3 = elephant herd walking at sunset]'"
    )
    image_prompt: str = Field(
        ...,
        description="Detailed, specific prompt for AI image generation. "
                    "Must describe scene, composition, mood, lighting, style, and subject clearly."
    )
    scene_description: str = Field(
        ...,
        description="Brief human-readable description of what this image should depict"
    )


class StepImageAllocation(BaseModel):
    """Image allocation for a single step — Phase 1 output."""
    step_number: int = Field(..., description="Step number this allocation applies to")
    step_description: str = Field(..., description="Brief description of what this step covers")
    allocated_images: int = Field(..., description="Number of images allocated to this step")
    image_hints: List[str] = Field(
        ...,
        description="Rough visual subject suggestions for each image in this step. "
                    "One hint per allocated image."
    )


class ImagePlanOutput(BaseModel):
    """Complete image plan — combines Phase 1 allocations + Phase 2 placements."""
    total_images: int = Field(..., description="Total number of images across all steps")
    default_images: int = Field(default=15, description="Default target image count")
    step_allocations: List[StepImageAllocation] = Field(
        ..., description="Image allocation per step (Phase 1 output)"
    )
    image_placements: List[ImagePlacement] = Field(
        ..., description="Detailed image placements and prompts (Phase 2 output)"
    )


class PerStepImageResult(BaseModel):
    """Structured output from a single per-step image placement call — Phase 2."""
    step_number: int = Field(..., description="Step number")
    annotated_step_text: str = Field(
        ...,
        description="The step's text with [image-N = visual description] markers "
                    "inserted at natural break points"
    )
    images: List[ImagePlacement] = Field(
        ..., description="Image placements for this step with detailed generation prompts"
    )


class StepImageAllocationList(BaseModel):
    """Wrapper for structured output of image allocations (Phase 1).
    Required because with_structured_output expects a single model, not a list.
    """
    allocations: List[StepImageAllocation] = Field(..., description="Image allocation for each step")


class GeneratedImageURL(BaseModel):
    """Maps an image_number to its live web URL after generation + upload."""
    image_number: int = Field(
        ..., description="The sequential image number matching ImagePlacement.image_number"
    )
    image_prompt: str = Field(
        ..., description="The prompt used to generate this image"
    )
    image_url: str = Field(
        ..., description="Live web URL of the generated image (hosted on ImgBB)"
    )
    status: str = Field(
        ..., description="'success' or 'failed'"
    )
    error: Optional[str] = Field(
        None, description="Error message if generation failed"
    )


class VideoDataResponse(BaseModel):
    """Complete API response for the /fetch-video-data endpoint."""
    # ── Original fetched data ──
    video_id: str
    title: str
    description: str
    tags: List[str]
    hashtags: str
    transcript: str
    thumbnail_url: Optional[str]
    # ── LangGraph rewritten data ──
    rewritten_title: Optional[str]
    rewritten_description: Optional[str]
    rewritten_hashtags: Optional[str]
    # ── Structured script steps ──
    script_steps: Optional[TranscriptStepsOutput] = None
    # ── Final polished YouTube script ──
    final_script: Optional[str] = None
    # ── Image planning ──
    image_plan: Optional[ImagePlanOutput] = None
    annotated_transcript: Optional[str] = None
    # ── Generated image URLs ──
    generated_image_urls: Optional[List[GeneratedImageURL]] = None
    # ── Generated thumbnail image URL (regenerated via Groq Vision + Cloudflare Worker) ──
    generated_thumbnail_url: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: LANGGRAPH STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class MetadataState(TypedDict):
    """Shared state flowing through the LangGraph pipeline.

    Complex/nested objects (TranscriptStepsOutput, ImagePlanOutput, etc.)
    are stored as JSON strings for LangGraph state compatibility.

    Per-request configuration (language, min_script_word_count, default_image_count)
    is stored here so each concurrent request has its own values.
    """
    # ── Per-request configuration ──
    language: str                                    # Target language for all generated content
    min_script_word_count: int                       # Minimum word count target for the script
    default_image_count: int                         # Default target number of images

    # ── Original fetched data ──
    title: str
    description: str
    hashtags: str
    thumbnail_url: Optional[str]
    transcript: Optional[str]

    # ── Rewritten metadata ──
    rewritten_title: Optional[str]
    rewritten_description: Optional[str]
    rewritten_hashtags: Optional[str]

    # ── Script planning ──
    # JSON string of TranscriptStepsOutput
    script_steps: Optional[str]

    # ── Script writing ──
    # Concatenated draft (all step outputs joined)
    written_steps_draft: Optional[str]
    # JSON array of per-step written texts
    written_steps_list: Optional[str]

    # ── Final polished script ──
    final_script: Optional[str]

    # ── Image planning ──
    # JSON of List[StepImageAllocation] (Phase 1)
    image_allocations: Optional[str]
    # JSON of ImagePlanOutput (combined Phase 1 + 2)
    image_plan: Optional[str]
    # Final transcript with [image-N] placeholders
    annotated_transcript: Optional[str]

    # ── Image generation ──
    # JSON of List[GeneratedImageURL]
    generated_image_urls: Optional[str]

    # ── Thumbnail analysis ──
    # Prompt from Groq Vision used to regenerate the thumbnail
    thumbnail_prompt: Optional[str]
    # Generated thumbnail image URL (from Cloudflare Worker + ImgBB)
    generated_thumbnail_url: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: LLM INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Key assignment strategy (balances load across both Gemini keys):
#   GROQ_API_KEY_1   → metadata_rewriter_llm  (3 calls: title, description, hashtags)
#                    + thumbnail_analyzer      (1 call: Groq Vision thumbnail analysis)
#   GEMINI_API_KEY_1 → transcript_steps_llm   (1 call: step planning)
#                    + script_polish_llm      (1 call: final polish)
#                    + image_placer_llm       (6-10 calls: per-step image placement)
#   GEMINI_API_KEY_2 → script_writer_llm      (6-10 calls: sequential step writing — heaviest)
#                    + image_allocator_llm    (1 call: image allocation)
#   CF_WORKER        → image_generator_node    (15+1 calls: 15 script images + 1 thumbnail)

metadata_rewriter_llm = ChatOpenAI(
    api_key=settings.gemini_api_key_1,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-2.5-flash",
)

# ── Gemini LLMs on KEY 1 ──
transcript_steps_llm = ChatGroq(
    api_key=settings.groq_api_key_3,
    model=settings.groq_model_name
)
script_polish_llm = ChatGroq(
    api_key=settings.groq_api_key_1,
    model=settings.groq_model_name
)
image_placer_llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model="gpt-4.1-nano",
)

# ── Gemini LLMs on KEY 2 ──
script_writer_llm = ChatOpenAI(
    api_key=settings.openai_api_key,
    model="gpt-4.1-nano",
)
image_allocator_llm = ChatGroq(
    api_key=settings.groq_api_key_2,
    model=settings.groq_model_name
)

print("[LLM_INIT][OK] All LLMs initialized (Groq_1 + Gemini_1 + Gemini_2).")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: LLM RETRY UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def invoke_with_retry(llm, messages, max_retries: int = 5, initial_delay: float = 6.0,
                      node_name: str = "unknown") -> object:
    """Invoke an LLM with exponential backoff retry for rate-limit (429) errors.
    Returns the LLM response on success, raises on non-retryable or exhausted errors.
    """
    retry_delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
            if is_rate_limit and attempt < max_retries:
                print(f"[RETRY][{node_name}] Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: LLM PROMPT TEMPLATES (Language-Aware)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Language Directive ────────────────────────────────────────────────────────
# Appended to every content-generating system prompt to enforce output language.

LANGUAGE_DIRECTIVE = """
LANGUAGE: Write ALL output in {language}. No exceptions."""

# ─── Node 1-3: Metadata Rewriter Prompts ──────────────────────────────────────

REWRITE_TITLE_SYSTEM = """Rewrite the title: 3-4% change, SEO friendly, max 80 chars.
Return ONLY the new title.""" + LANGUAGE_DIRECTIVE

REWRITE_TITLE_USER = "Title: {title}"


REWRITE_DESCRIPTION_SYSTEM = """Rewrite the description: 3-4% change, SEO optimized.
Return ONLY the new description.""" + LANGUAGE_DIRECTIVE

REWRITE_DESCRIPTION_USER = "Description: {description}"


REWRITE_HASHTAGS_SYSTEM = """Generate 5 hashtags on the same topic.
Format: #tag1, #tag2, #tag3, #tag4, #tag5
Return ONLY the hashtags.""" + LANGUAGE_DIRECTIVE

REWRITE_HASHTAGS_USER = "Original hashtags: {hashtags}"


# ─── Node 4: Transcript Steps Maker ───────────────────────────────────────────

TRANSCRIPT_STEPS_SYSTEM = """You are a script planner.
Break the transcript into 3 steps ONLY (for testing).
Each step MUST have:
- step_number, description (1 sentence), continuity_note (1 sentence), tone, word_count (set to 50)
Total word count across steps: ~150 words.
Return result matching TranscriptStepsOutput schema.
Write in {language}.""" + LANGUAGE_DIRECTIVE

TRANSCRIPT_STEPS_USER = "Transcript:\n{transcript}"


# ─── Node 5: Script Writer ────────────────────────────────────────────────────

SCRIPT_WRITER_SYSTEM = "You are a YouTube script writer." + LANGUAGE_DIRECTIVE

SCRIPT_WRITER_USER = """Write Step {step_number} of {total_steps} for: {video_title}

- Description: {description}
- Tone: {tone}
- Word count: {word_count} words MAX (testing mode — keep it short)
- Continue naturally from: \"{prev_output}\"
- Use facts from: \"{transcript}\"

Rules: Script text only. No headers. Max {word_count} words. Write in {language}."""


# ─── Node 6: Script Polish ────────────────────────────────────────────────────

SCRIPT_POLISH_SYSTEM = "You are a script editor." + LANGUAGE_DIRECTIVE

SCRIPT_POLISH_USER = """Polish this script for: {video_title}

Rules:
1. Smooth transitions only — do NOT rewrite content.
2. Remove repetitions.
3. Keep it under 200 words (testing mode).
4. Return ONLY the script. No commentary.
5. Write in {language}.

SCRIPT:
\"\"\"{draft}\"\"\"
"""


# ─── Node 7: Image Allocator ───────────────────────────────────────────────────

IMAGE_ALLOCATOR_SYSTEM = """You are a video image planner.
Allocate exactly 1 image per step (testing mode).
For each step provide 1 short visual hint (max 5 words).
Return result matching StepImageAllocationList schema.
Write step_description and image_hints in English."""

IMAGE_ALLOCATOR_USER = "Title: {video_title}\nSteps:\n{steps_summary}"


# ─── Node 8: Image Placer ──────────────────────────────────────────────────────

IMAGE_PLACER_SYSTEM = "You are a video image director."

IMAGE_PLACER_USER = """Place {image_count} image(s) in this script step.
Title: {video_title} | Step {step_number}: {step_description}
Image numbers: {image_nums_str}
Hints: {hints_str}

STEP TEXT:
\"\"\"{step_text}\"\"\"

Rules:
1. Insert [image-N = short description max 8 words] at a natural break.
2. image_prompt: max 20 words, English only, 16:9.
3. Keep step text in {language}.
Return PerStepImageResult schema."""


# ─── Node 9: Thumbnail Analyzer ───────────────────────────────────────────────

THUMBNAIL_ANALYSIS_PROMPT = """Analyze this YouTube thumbnail.
Write ONE image generation prompt that recreates it with 3-4% variation.
Keep: layout, text, style, composition.
Max 50 words.
Return ONLY the prompt."""
# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: YOUTUBE DATA FETCHING SERVICES
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Language Name → YouTube Transcript Language Code Mapping ──────────────────
from typing import List

LANGUAGE_CODE_MAP = {
    "english": "en",
    "en": "en",
    "urdu": "ur",
    "hindi": "hi",
    "arabic": "ar",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "russian": "ru",
    "japanese": "ja",
    "korean": "ko",
    "chinese": "zh",
    "chinese simplified": "zh-Hans",
    "chinese traditional": "zh-Hant",
    "turkish": "tr",
    "persian": "fa",
    "bengali": "bn",
    "indonesian": "id",
    "italian": "it",
    "dutch": "nl",
    "polish": "pl",
    "thai": "th",
    "vietnamese": "vi",
    "malay": "ms",
    "tamil": "ta",
    "telugu": "te",
    "punjabi": "pa",
    "marathi": "mr",
    "swedish": "sv",
    "norwegian": "no",
    "danish": "da",
    "finnish": "fi",
    "greek": "el",
    "czech": "cs",
    "romanian": "ro",
    "hungarian": "hu",
    "ukrainian": "uk",
    "hebrew": "he",
}

# Regional variants YouTube actually uses per primary code
LANGUAGE_VARIANTS = {
    "en":      ["en", "en-US", "en-GB", "en-IN", "en-AU", "en-CA"],
    "hi":      ["hi", "hi-IN"],
    "ur":      ["ur", "ur-PK", "ur-IN"],
    "zh":      ["zh-Hans", "zh-Hant", "zh-CN", "zh-TW", "zh-HK"],
    "pt":      ["pt", "pt-BR", "pt-PT"],
    "es":      ["es", "es-ES", "es-MX", "es-419", "es-US"],
    "fr":      ["fr", "fr-FR", "fr-CA", "fr-BE"],
    "ar":      ["ar", "ar-SA", "ar-EG"],
    "de":      ["de", "de-DE", "de-AT", "de-CH"],
    "bn":      ["bn", "bn-IN", "bn-BD"],
    "pa":      ["pa", "pa-IN", "pa-PK"],
    "ta":      ["ta", "ta-IN", "ta-LK"],
    "te":      ["te", "te-IN"],
    "ms":      ["ms", "ms-MY"],
    "ko":      ["ko", "ko-KR"],
    "ja":      ["ja", "ja-JP"],
    "ru":      ["ru", "ru-RU"],
    "tr":      ["tr", "tr-TR"],
    "nl":      ["nl", "nl-NL", "nl-BE"],
    "id":      ["id", "id-ID"],
    "vi":      ["vi", "vi-VN"],
    "fa":      ["fa", "fa-IR"],
    "th":      ["th", "th-TH"],
    "pl":      ["pl", "pl-PL"],
    "it":      ["it", "it-IT"],
    "sv":      ["sv", "sv-SE"],
    "no":      ["no", "nb", "nb-NO"],
    "da":      ["da", "da-DK"],
    "fi":      ["fi", "fi-FI"],
    "el":      ["el", "el-GR"],
    "cs":      ["cs", "cs-CZ"],
    "ro":      ["ro", "ro-RO"],
    "hu":      ["hu", "hu-HU"],
    "uk":      ["uk", "uk-UA"],
    "he":      ["he", "iw"],        # YouTube uses both "he" and legacy "iw"
}

# Broad fallback chain — ordered by global speaker count
GLOBAL_FALLBACK_CHAIN = [
    "en", "en-US", "en-GB", "en-IN",          # English
    "hi", "hi-IN",                              # Hindi
    "zh-Hans", "zh-Hant",                       # Chinese
    "es", "es-MX",                              # Spanish
    "ar",                                       # Arabic
    "pt", "pt-BR",                              # Portuguese
    "fr",                                       # French
    "ur", "ur-PK",                              # Urdu
    "bn", "bn-IN",                              # Bengali
    "ru",                                       # Russian
    "de",                                       # German
    "ja",                                       # Japanese
    "id",                                       # Indonesian
    "ko",                                       # Korean
    "tr",                                       # Turkish
    "vi",                                       # Vietnamese
    "it",                                       # Italian
    "ms",                                       # Malay
    "fa",                                       # Persian
    "ta", "ta-IN",                              # Tamil
    "te", "pa", "mr",                           # South Asian
    "nl", "pl", "th", "sv", "no", "da",        # European + Thai
    "fi", "el", "cs", "ro", "hu", "uk", "he",  # European
]


def _get_language_codes(language: str) -> List[str]:
    """Convert a language name to YouTube transcript language codes.
    Returns a list of codes to try, with English and Hindi as fallbacks.

    Priority order:
      1. Requested language + all its known regional variants
      2. English variants     (skipped if English was requested)
      3. Hindi variants       (skipped if Hindi was requested)
      4. Global fallback chain covering 30+ major languages
    """
    lang_lower = language.lower().strip()
    primary_code = LANGUAGE_CODE_MAP.get(lang_lower, None)

    seen = set()
    codes = []

    def _add(code: str) -> None:
        if code not in seen:
            seen.add(code)
            codes.append(code)

    # 1. Primary requested language + all its regional variants
    if primary_code:
        for variant in LANGUAGE_VARIANTS.get(primary_code, [primary_code]):
            _add(variant)
        _add(primary_code)  # safety: add bare code if not in LANGUAGE_VARIANTS

    # 2. English fallback (skip if English was already requested)
    if primary_code != "en":
        for variant in LANGUAGE_VARIANTS["en"]:
            _add(variant)

    # 3. Hindi fallback (skip if Hindi was already requested)
    if primary_code != "hi":
        for variant in LANGUAGE_VARIANTS["hi"]:
            _add(variant)

    # 4. Global fallback chain
    for code in GLOBAL_FALLBACK_CHAIN:
        _add(code)

    return codes

def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    print(f"[EXTRACT_VIDEO_ID] Extracting from: {url}")
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    if match:
        video_id = match.group(1)
        print(f"[EXTRACT_VIDEO_ID][OK] {video_id}")
        return video_id
    print("[EXTRACT_VIDEO_ID][ERROR] Could not extract video ID.")
    return None


def fetch_metadata(video_id: str) -> Optional[dict]:
    """Fetch video metadata (title, description, tags) via YouTube Data API."""
    print(f"[FETCH_METADATA] Fetching for: {video_id}")
    try:
        youtube = build("youtube", "v3", developerKey=settings.youtube_api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()
        if not response.get("items"):
            print("[FETCH_METADATA][ERROR] No items found.")
            return None
        item = response["items"][0]["snippet"]
        metadata = {
            "title": item["title"],
            "description": item["description"],
            "tags": item.get("tags", [])
        }
        print(f"[FETCH_METADATA][OK] Title: {metadata['title']}")
        return metadata
    except HttpError as e:
        print(f"[FETCH_METADATA][ERROR] HttpError: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[FETCH_METADATA][ERROR] {e}")
        traceback.print_exc()
        return None


def build_metadata_string(metadata: dict) -> str:
    """Build a comma-separated hashtag string from metadata tags and description."""
    print("[BUILD_METADATA_STRING] Building hashtags...")
    tags = metadata.get("tags", [])
    description = metadata["description"]
    hashtags_from_desc = re.findall(r"#\w+", description)
    hashtags_from_tags = ["#" + t.replace(" ", "") for t in tags]
    all_hashtags = ', '.join(list(set(hashtags_from_desc + hashtags_from_tags)))
    print(f"[BUILD_METADATA_STRING][OK] {all_hashtags[:80]}")
    return all_hashtags


def fetch_transcript(video_id: str, language: str = "English") -> Optional[str]:
    """Fetch video transcript via youtube_transcript_api.

    Tries to fetch in the specified language first. Falls back to English
    if the requested language transcript is not available.
    The transcript text itself will be in the language available on YouTube;
    the LLM pipeline will translate/rewrite it into the target language.
    """
    language_codes = _get_language_codes(language)
    print(f"[FETCH_TRANSCRIPT] Fetching for: {video_id} (language: {language}, codes: {language_codes})")

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.fetch(video_id, languages=language_codes)
        full_text = " ".join([snippet.text for snippet in transcript_list])
        result = full_text.strip()
        print(f"[FETCH_TRANSCRIPT][OK] Length: {len(result)} chars (fetched in {language_codes[0]})")
        return result
    except Exception as e:
        print(f"[FETCH_TRANSCRIPT][ERROR] {e}")
        traceback.print_exc()
        return None


def fetch_thumbnail_url(url: str) -> Optional[str]:
    """Extract video ID and return the best available thumbnail URL.
    Tries multiple YouTube thumbnail resolutions in order of quality.
    Old videos often lack maxresdefault, so we fallback.
    """
    print(f"[FETCH_THUMBNAIL] Fetching for: {url}")
    try:
        parsed_url = urlparse(url)
        video_id = None
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        elif parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path.lstrip('/')
        if not video_id:
            print("[FETCH_THUMBNAIL][ERROR] Could not extract video ID.")
            return None

        # Try thumbnail resolutions from highest to lowest quality
        thumbnail_resolutions = [
            "maxresdefault",  # 1280x720
            "sddefault",      # 640x480
            "hqdefault",      # 480x360
            "mqdefault",      # 320x180
            "default",        # 120x90
        ]

        for res in thumbnail_resolutions:
            candidate = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
            try:
                with httpx.Client(timeout=10) as client:
                    resp = client.head(candidate)
                    if resp.status_code == 200:
                        content_length = resp.headers.get("content-length")
                        if content_length and int(content_length) > 2000:
                            print(f"[FETCH_THUMBNAIL][OK] {candidate} ({res})")
                            return candidate
                        elif not content_length:
                            print(f"[FETCH_THUMBNAIL][OK] {candidate} ({res})")
                            return candidate
                        else:
                            print(f"[FETCH_THUMBNAIL][SKIP] {candidate} — placeholder ({content_length} bytes)")
                    else:
                        print(f"[FETCH_THUMBNAIL][SKIP] {candidate} — HTTP {resp.status_code}")
            except Exception as e:
                print(f"[FETCH_THUMBNAIL][SKIP] {candidate} — {e}")
                continue

        # Fallback: return hqdefault even if it might be a placeholder
        fallback = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
        print(f"[FETCH_THUMBNAIL][FALLBACK] {fallback}")
        return fallback
    except Exception as e:
        print(f"[FETCH_THUMBNAIL][ERROR] {e}")
        traceback.print_exc()
        return None


def validate_thumbnail_url(thumbnail_url: str) -> Optional[str]:
    """Validate that a thumbnail URL actually returns an image.
    Returns the URL if valid, None otherwise.
    Also tries fallback resolutions if the original URL fails.
    """
    if not thumbnail_url:
        return None

    # If it's a YouTube img URL, we can try fallback resolutions
    yt_img_pattern = r"https://img\.youtube\.com/vi/([a-zA-Z0-9_-]+)/(.+)"
    match = re.match(yt_img_pattern, thumbnail_url)

    urls_to_try = [thumbnail_url]

    if match:
        video_id = match.group(1)
        current_res = match.group(2).replace(".jpg", "")
        all_resolutions = ["maxresdefault", "sddefault", "hqdefault", "mqdefault", "default"]
        try:
            current_idx = all_resolutions.index(current_res)
            for res in all_resolutions[current_idx + 1:]:
                urls_to_try.append(f"https://img.youtube.com/vi/{video_id}/{res}.jpg")
        except ValueError:
            pass

    for url in urls_to_try:
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.head(url)
                if resp.status_code == 200:
                    content_length = resp.headers.get("content-length")
                    if content_length and int(content_length) < 2000:
                        print(f"[THUMBNAIL_VALIDATE][SKIP] {url} — placeholder ({content_length} bytes)")
                        continue
                    print(f"[THUMBNAIL_VALIDATE][OK] {url}")
                    return url
                else:
                    print(f"[THUMBNAIL_VALIDATE][SKIP] {url} — HTTP {resp.status_code}")
        except Exception as e:
            print(f"[THUMBNAIL_VALIDATE][SKIP] {url} — {e}")
            continue

    print("[THUMBNAIL_VALIDATE][ERROR] No valid thumbnail URL found.")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: IMAGE GENERATION SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_single_image(prompt: str) -> Dict:
    """Generate a single image via Cloudflare Worker and upload to ImgBB.
    Returns a dict with keys: image_url, status, error.
    Uses synchronous httpx.Client (compatible with sync LangGraph nodes).
    """
    try:
        with httpx.Client(timeout=60) as client:
            # Step 1: Generate image via Cloudflare Worker
            cf_headers = {
                "Authorization": f"Bearer {settings.cf_worker_api_key}",
                "Content-Type": "application/json"
            }
            cf_payload = {
                "prompt": prompt,
                "model": settings.cf_model
            }
            cf_response = client.post(settings.cf_worker_url, headers=cf_headers, json=cf_payload)
            cf_response.raise_for_status()
            image_bytes = cf_response.content

            # Step 2: Upload raw bytes to ImgBB
            imgbb_params = {"key": settings.imgbb_api_key}
            imgbb_files = {"image": ("image.jpg", image_bytes)}
            imgbb_response = client.post(
                "https://api.imgbb.com/1/upload",
                params=imgbb_params,
                files=imgbb_files
            )
            imgbb_response.raise_for_status()

            # Step 3: Extract live URL
            result = imgbb_response.json()
            live_url = result["data"]["url"]

            return {
                "image_url": live_url,
                "status": "success",
                "error": None
            }

    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        print(f"[IMAGE_GEN][HTTP_ERROR] {error_msg}")
        return {
            "image_url": "",
            "status": "failed",
            "error": error_msg
        }
    except Exception as e:
        error_msg = str(e)[:200]
        print(f"[IMAGE_GEN][ERROR] {error_msg}")
        return {
            "image_url": "",
            "status": "failed",
            "error": error_msg
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: LANGGRAPH NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Node 1: Rewrite Title ────────────────────────────────────────────────────

def rewrite_title_node(state: MetadataState) -> dict:
    """Rewrite the video title with 3-4% variation in the target language."""
    print("[LANGGRAPH][NODE] rewrite_title running...")

    language = state.get("language", "English")

    try:
        messages = [
            SystemMessage(content=REWRITE_TITLE_SYSTEM.format(language=language)),
            HumanMessage(content=REWRITE_TITLE_USER.format(title=state['title']))
        ]
        response = invoke_with_retry(metadata_rewriter_llm, messages, node_name="rewrite_title")
        rewritten = response.content.strip()
        print(f"[LANGGRAPH][NODE][OK] rewrite_title ({language}): {rewritten}")
        return {"rewritten_title": rewritten}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] rewrite_title: {e}")
        traceback.print_exc()
        raise


# ─── Node 2: Rewrite Description ──────────────────────────────────────────────

def rewrite_description_node(state: MetadataState) -> dict:
    """Rewrite the video description with 3-4% variation in the target language."""
    print("[LANGGRAPH][NODE] rewrite_description running...")

    language = state.get("language", "English")

    try:
        messages = [
            SystemMessage(content=REWRITE_DESCRIPTION_SYSTEM.format(language=language)),
            HumanMessage(content=REWRITE_DESCRIPTION_USER.format(description=state['description']))
        ]
        response = invoke_with_retry(metadata_rewriter_llm, messages, node_name="rewrite_description")
        rewritten = response.content.strip()
        print(f"[LANGGRAPH][NODE][OK] rewrite_description ({language}). Length: {len(rewritten)} chars")
        return {"rewritten_description": rewritten}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] rewrite_description: {e}")
        traceback.print_exc()
        raise


# ─── Node 3: Rewrite Hashtags ─────────────────────────────────────────────────

def rewrite_hashtags_node(state: MetadataState) -> dict:
    """Rewrite hashtags with 3-4% variation in the target language."""
    print("[LANGGRAPH][NODE] rewrite_hashtags running...")

    language = state.get("language", "English")

    try:
        messages = [
            SystemMessage(content=REWRITE_HASHTAGS_SYSTEM.format(language=language)),
            HumanMessage(content=REWRITE_HASHTAGS_USER.format(hashtags=state['hashtags']))
        ]
        response = invoke_with_retry(metadata_rewriter_llm, messages, node_name="rewrite_hashtags")
        rewritten = response.content.strip()
        print(f"[LANGGRAPH][NODE][OK] rewrite_hashtags ({language}): {rewritten}")
        return {"rewritten_hashtags": rewritten}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] rewrite_hashtags: {e}")
        traceback.print_exc()
        raise


# ─── Node 4: Transcript Steps Maker ───────────────────────────────────────────

def transcript_steps_maker_node(state: MetadataState) -> dict:
    """Break transcript into structured script steps."""
    print("[LANGGRAPH][NODE] transcript_steps_maker running...")

    language = state.get("language", "English")
    min_word_count = state.get("min_script_word_count", 2500)

    try:
        messages = [
            SystemMessage(content=TRANSCRIPT_STEPS_SYSTEM.format(
                min_word_count=min_word_count,
                language=language
            )),
            HumanMessage(content=TRANSCRIPT_STEPS_USER.format(transcript=state['transcript']))
        ]

        structured_llm = transcript_steps_llm.with_structured_output(TranscriptStepsOutput)
        result: TranscriptStepsOutput = invoke_with_retry(
            structured_llm, messages, node_name="transcript_steps_maker"
        )
        script_steps_json = result.model_dump_json()
        print(f"[LANGGRAPH][NODE][OK] transcript_steps_maker ({language}). Steps: {result.total_steps}")
        return {"script_steps": script_steps_json}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] transcript_steps_maker: {e}")
        traceback.print_exc()
        raise


# ─── Node 5: Script Writer ────────────────────────────────────────────────────

def script_writer_node(state: MetadataState) -> dict:
    """Write script sequentially, one step at a time."""
    print("[LANGGRAPH][NODE] script_writer running...")

    raw_steps = state.get("script_steps")
    if not raw_steps:
        print("[LANGGRAPH][NODE][ERROR] script_writer: No script_steps found in state.")
        return {"written_steps_draft": None, "written_steps_list": None}

    try:
        steps_output: TranscriptStepsOutput = TranscriptStepsOutput.model_validate_json(raw_steps)
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] script_writer: Could not parse script_steps: {e}")
        return {"written_steps_draft": None, "written_steps_list": None}

    language = state.get("language", "English")
    min_word_count = state.get("min_script_word_count", 2500)
    transcript = state.get("transcript", "")
    video_title = state.get("title", "")

    # Word count scaling safety net
    planned_total = steps_output.total_word_count
    if planned_total < min_word_count:
        scale_factor = min_word_count / planned_total
        print(f"[LANGGRAPH][NODE][SCALE] Planned {planned_total} words < {min_word_count} minimum. "
              f"Scaling by {scale_factor:.2f}x")
        scaled_steps = []
        for step in steps_output.steps:
            new_wc = max(int(step.word_count * scale_factor), 200)
            scaled_steps.append(Step(
                step_number=step.step_number,
                description=step.description,
                continuity_note=step.continuity_note,
                tone=step.tone,
                word_count=new_wc
            ))
        steps_output = TranscriptStepsOutput(
            video_title=steps_output.video_title,
            total_steps=steps_output.total_steps,
            total_word_count=sum(s.word_count for s in scaled_steps),
            steps=scaled_steps
        )
        print(f"[LANGGRAPH][NODE][SCALE] New total: {steps_output.total_word_count} words")

    written_steps: List[str] = []

    for step in steps_output.steps:
        prev_output = written_steps[-1] if written_steps else "This is the first step — no previous output exists yet."

        prompt = SCRIPT_WRITER_USER.format(
            video_title=video_title,
            step_number=step.step_number,
            total_steps=steps_output.total_steps,
            description=step.description,
            continuity_note=step.continuity_note,
            tone=step.tone,
            word_count=step.word_count,
            prev_output=prev_output,
            transcript=transcript,
            language=language
        )

        try:
            messages = [
                SystemMessage(content=SCRIPT_WRITER_SYSTEM.format(language=language)),
                HumanMessage(content=prompt)
            ]
            response = invoke_with_retry(
                script_writer_llm, messages,
                node_name=f"script_writer_step_{step.step_number}"
            )
            step_text = response.content.strip()
            written_steps.append(step_text)

            word_count = len(step_text.split())
            print(f"[LANGGRAPH][NODE][OK] script_writer step {step.step_number}: "
                  f"{word_count} words (target: {step.word_count}) [{language}]")
        except Exception as e:
            print(f"[LANGGRAPH][NODE][ERROR] script_writer step {step.step_number}: {e}")
            traceback.print_exc()
            raise

    draft = "\n\n".join(written_steps)
    total_words = sum(len(s.split()) for s in written_steps)
    print(f"[LANGGRAPH][NODE][OK] script_writer complete: {len(written_steps)} steps, ~{total_words} total words [{language}]")
    return {"written_steps_draft": draft, "written_steps_list": json.dumps(written_steps)}


# ─── Node 6: Script Polish ────────────────────────────────────────────────────

def script_polish_node(state: MetadataState) -> dict:
    """Polish the concatenated draft into one cohesive script."""
    print("[LANGGRAPH][NODE] script_polish running...")

    draft = state.get("written_steps_draft")
    if not draft:
        print("[LANGGRAPH][NODE][ERROR] script_polish: No written_steps_draft found.")
        return {"final_script": None}

    language = state.get("language", "English")
    min_word_count = state.get("min_script_word_count", 2500)
    video_title = state.get("title", "")
    draft_word_count = len(draft.split())

    prompt = SCRIPT_POLISH_USER.format(
        video_title=video_title,
        draft_word_count=draft_word_count,
        min_word_count=min_word_count,
        draft=draft,
        language=language
    )

    try:
        messages = [
            SystemMessage(content=SCRIPT_POLISH_SYSTEM.format(language=language)),
            HumanMessage(content=prompt)
        ]
        response = invoke_with_retry(script_polish_llm, messages, node_name="script_polish")
        polished = response.content.strip()

        total_words = len(polished.split())
        print(f"[LANGGRAPH][NODE][OK] script_polish ({language}): Final script ~{total_words} words")
        return {"final_script": polished}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] script_polish: {e}")
        traceback.print_exc()
        raise


# ─── Node 7: Image Allocator (Phase 1) ────────────────────────────────────────

def image_allocator_node(state: MetadataState) -> dict:
    """Phase 1: Decide how many images per step."""
    print("[LANGGRAPH][NODE] image_allocator (Phase 1) running...")

    raw_steps = state.get("script_steps")
    if not raw_steps:
        print("[LANGGRAPH][NODE][ERROR] image_allocator: No script_steps found.")
        return {"image_allocations": None}

    try:
        steps_output: TranscriptStepsOutput = TranscriptStepsOutput.model_validate_json(raw_steps)
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] image_allocator: Could not parse script_steps: {e}")
        return {"image_allocations": None}

    default_image_count = state.get("default_image_count", 15)

    # Build compact step summary
    steps_summary = ""
    for step in steps_output.steps:
        steps_summary += (
            f"Step {step.step_number}: {step.description} "
            f"(Tone: {step.tone}, Word Count: {step.word_count})\n"
        )

    messages = [
        SystemMessage(content=IMAGE_ALLOCATOR_SYSTEM.format(
            default_image_count=default_image_count
        )),
        HumanMessage(content=IMAGE_ALLOCATOR_USER.format(
            video_title=state.get("title", ""),
            steps_summary=steps_summary
        ))
    ]

    try:
        structured_llm = image_allocator_llm.with_structured_output(StepImageAllocationList)
        result: StepImageAllocationList = invoke_with_retry(
            structured_llm, messages, node_name="image_allocator"
        )

        allocations_json = json.dumps([a.model_dump() for a in result.allocations])
        total = sum(a.allocated_images for a in result.allocations)
        print(f"[LANGGRAPH][NODE][OK] image_allocator: {total} images across {len(result.allocations)} steps")
        return {"image_allocations": allocations_json}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] image_allocator: {e}")
        traceback.print_exc()
        raise


# ─── Node 8: Image Placer (Phase 2) ───────────────────────────────────────────

def image_placer_node(state: MetadataState) -> dict:
    """Phase 2: Place images with detailed prompts in each step."""
    print("[LANGGRAPH][NODE] image_placer (Phase 2) running...")

    raw_allocations = state.get("image_allocations")
    raw_steps_list = state.get("written_steps_list")

    if not raw_allocations or not raw_steps_list:
        print("[LANGGRAPH][NODE][ERROR] image_placer: Missing allocations or steps_list.")
        return {"image_plan": None, "annotated_transcript": None}

    try:
        allocations: List[StepImageAllocation] = [
            StepImageAllocation(**a) for a in json.loads(raw_allocations)
        ]
        step_texts: List[str] = json.loads(raw_steps_list)
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] image_placer: Could not parse inputs: {e}")
        return {"image_plan": None, "annotated_transcript": None}

    language = state.get("language", "English")
    default_image_count = state.get("default_image_count", 15)
    video_title = state.get("title", "")
    all_image_placements: List = []
    all_annotated_steps: List[str] = []
    global_image_counter = 0

    for alloc in allocations:
        step_idx = alloc.step_number - 1
        if step_idx < 0 or step_idx >= len(step_texts):
            print(f"[LANGGRAPH][NODE][WARN] image_placer: Step {alloc.step_number} out of range, skipping.")
            continue

        step_text = step_texts[step_idx]
        image_count = alloc.allocated_images

        start_image_num = global_image_counter + 1
        end_image_num = global_image_counter + image_count
        image_nums_str = ", ".join(str(n) for n in range(start_image_num, end_image_num + 1))

        hints_str = "\n".join(
            f"  Image {start_image_num + i}: {hint}"
            for i, hint in enumerate(alloc.image_hints)
        )

        prompt = IMAGE_PLACER_USER.format(
            video_title=video_title,
            step_number=alloc.step_number,
            step_description=alloc.step_description,
            image_count=image_count,
            image_nums_str=image_nums_str,
            hints_str=hints_str,
            step_text=step_text,
            language=language
        )

        # Retry with exponential backoff for rate-limit errors
        MAX_RETRIES = 5
        retry_delay = 6
        step_success = False

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                messages = [
                    SystemMessage(content=IMAGE_PLACER_SYSTEM),
                    HumanMessage(content=prompt)
                ]
                structured_llm = image_placer_llm.with_structured_output(PerStepImageResult)
                result: PerStepImageResult = invoke_with_retry(
                    structured_llm, messages,
                    node_name=f"image_placer_step_{alloc.step_number}"
                )

                all_annotated_steps.append(result.annotated_step_text)
                all_image_placements.extend(result.images)
                print(f"[LANGGRAPH][NODE][OK] image_placer step {alloc.step_number}: "
                      f"{len(result.images)} images placed")
                step_success = True
                break

            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "rate_limit" in error_str.lower()
                if is_rate_limit and attempt < MAX_RETRIES:
                    print(f"[LANGGRAPH][NODE][RETRY] image_placer step {alloc.step_number}: "
                          f"Rate limit hit (attempt {attempt}/{MAX_RETRIES}). Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 30)
                else:
                    print(f"[LANGGRAPH][NODE][ERROR] image_placer step {alloc.step_number}: {e}")
                    traceback.print_exc()
                    all_annotated_steps.append(step_text)
                    break

        if not step_success:
            print(f"[LANGGRAPH][NODE][WARN] image_placer step {alloc.step_number}: "
                  f"Failed after {attempt} attempt(s). Continuing without images.")

        global_image_counter += image_count

    # Build final outputs
    allocation_objects = [StepImageAllocation(**a) for a in json.loads(raw_allocations)]
    image_plan = ImagePlanOutput(
        total_images=len(all_image_placements),
        default_images=default_image_count,
        step_allocations=allocation_objects,
        image_placements=all_image_placements
    )
    annotated_transcript = "\n\n".join(all_annotated_steps)
    image_plan_json = image_plan.model_dump_json()

    print(f"[LANGGRAPH][NODE][OK] image_placer complete: "
          f"{image_plan.total_images} images, annotated transcript ~{len(annotated_transcript.split())} words")
    return {"image_plan": image_plan_json, "annotated_transcript": annotated_transcript}


# ─── Node 9: Thumbnail Analyzer ────────────────────────────────────────────────

import traceback
import logging
# from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tenacity import retry, stop_after_attempt, wait_fixed, before_log, after_log

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THUMBNAIL_ANALYSIS_PROMPT = """
You are a world-class image analyst and professional AI image prompt engineer 
specializing in YouTube thumbnail reconstruction. You have a sharp eye for 
design details including exact colors, typography, spatial composition, 
lighting, and illustration style. Your analysis must be EXHAUSTIVE and PRECISE.

Carefully analyze this YouTube thumbnail image. Be extremely precise.

STEP 1 — DEEP VISUAL ANALYSIS:

BACKGROUND:
- Exact color(s) and hex approximation
- Gradient direction and colors (radial, top-bottom, diagonal)
- Any background shapes (circles, blobs, geometric forms) and their opacity

SUBJECTS & CHARACTERS:
- Number of people/characters/objects
- Exact position (left-back, right-front, center, overlapping)
- Height relationship (who is taller, who is in front)
- Clothing: exact colors, type, layers
- Physical features: hair color/style, eye color, expression
- Body language: arms crossed, pointing, standing
- Drop shadows or glow effects on subjects

TYPOGRAPHY (critical — observe every character):
- Every word/letter visible
- Font style: bold, italic, 3D, flat
- Font size relative to image: massive, large, medium, small
- Exact color of each word/letter
- Outline: color and thickness
- Shadow or 3D pop-out effect
- Position: bottom-left, top-center, overlapping subjects or not

COLOR PALETTE:
- List ALL dominant colors with descriptions
- Note color relationships between elements

LIGHTING & EFFECTS:
- Glow effects and their colors
- Radial or directional gradients
- Any special overlays or vignettes

STYLE:
- Flat vector cartoon / 3D render / photorealistic / illustrated
- Overall mood: professional, playful, dramatic, educational

STEP 2 — GENERATE THE IMAGE PROMPT:

Write ONE detailed image generation prompt that:
- Reconstructs this thumbnail with 3-4% variation ONLY
- Preserves exact layout, composition, and depth
- Keeps ALL visible text exactly as written
- Describes subjects from background to foreground
- Only shifts color tones by ~5%

STRICT FORMAT:
- ONE flowing paragraph, no bullet points, no headers
- Describe: background → back subjects → front subjects → text
- End EXACTLY with: sharp focus, 8k resolution, professional YouTube thumbnail design, high contrast
- Return ONLY the prompt. Nothing else.
"""


image_analyzer_llm = ChatGroq(
    api_key=settings.groq_api_key_2,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    max_tokens=2048,
    timeout=120,
    model_kwargs={"top_p": 0.85}   # ✅ correct way
)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    before=before_log(logger, logging.INFO),
    after=after_log(logger, logging.INFO),
    reraise=True
)

def _invoke_vision(image_url: str) -> str:
    print(f"[VISION] Calling LLM for: {image_url}")
    message = HumanMessage(
        content=[
            {"type": "text",      "text": THUMBNAIL_ANALYSIS_PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    response = image_analyzer_llm.invoke([message])
    return response.content.strip()


def thumbnail_analyzer_node(state: MetadataState) -> MetadataState:
    print("[LANGGRAPH][NODE] thumbnail_analyzer running...")

    thumbnail_url = state.get("thumbnail_url")

    if not thumbnail_url:
        print("[LANGGRAPH][NODE][WARN] No thumbnail_url in state. Skipping.")
        return {"thumbnail_prompt": None}

    print(f"[VISION] Analyzing thumbnail: {thumbnail_url}")  # ✅ fixed
    try:
        prompt = _invoke_vision(thumbnail_url)
        print(f"[LANGGRAPH][NODE][OK] Prompt generated ({len(prompt)} chars)")
        print(f"[LANGGRAPH][NODE][OK] {prompt}")
        return {"thumbnail_prompt": prompt}
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] {e}")
        traceback.print_exc()
        return {"thumbnail_prompt": None}

# ─── Node 10: Image Generator ─────────────────────────────────────────────────

def image_generator_node(state: MetadataState) -> dict:
    """Generate all images from the image_plan + thumbnail recreation."""
    print("[LANGGRAPH][NODE] image_generator running...")

    raw_image_plan = state.get("image_plan")
    if not raw_image_plan:
        print("[LANGGRAPH][NODE][ERROR] image_generator: No image_plan found in state.")
        return {"generated_image_urls": None, "generated_thumbnail_url": None}

    try:
        image_plan: ImagePlanOutput = ImagePlanOutput.model_validate_json(raw_image_plan)
    except Exception as e:
        print(f"[LANGGRAPH][NODE][ERROR] image_generator: Could not parse image_plan: {e}")
        return {"generated_image_urls": None, "generated_thumbnail_url": None}

    if not image_plan.image_placements:
        print("[LANGGRAPH][NODE][WARN] image_generator: No image_placements in image_plan.")
        return {"generated_image_urls": None, "generated_thumbnail_url": None}

    # Generate each image
    generated_urls: List[GeneratedImageURL] = []
    total = len(image_plan.image_placements)

    for i, placement in enumerate(image_plan.image_placements):
        print(f"[LANGGRAPH][NODE] image_generator: Generating image {placement.image_number}/{total} "
              f"(step {placement.step_number})...")

        result = generate_single_image(placement.image_prompt)

        generated_urls.append(GeneratedImageURL(
            image_number=placement.image_number,
            image_prompt=placement.image_prompt,
            image_url=result["image_url"],
            status=result["status"],
            error=result.get("error")
        ))

        if result["status"] == "success":
            print(f"[LANGGRAPH][NODE][OK] image_generator: Image {placement.image_number} → {result['image_url'][:60]}...")
        else:
            print(f"[LANGGRAPH][NODE][WARN] image_generator: Image {placement.image_number} FAILED — {result.get('error', 'unknown')}")

        if i < total - 1:
            time.sleep(settings.image_gen_delay_sec)

    # Generate thumbnail recreation image
    generated_thumbnail_url = None
    thumbnail_prompt = state.get("thumbnail_prompt")
    if thumbnail_prompt:
        print("[LANGGRAPH][NODE] image_generator: Generating regenerated thumbnail image...")
        time.sleep(settings.image_gen_delay_sec)

        thumb_result = generate_single_image(thumbnail_prompt)
        if thumb_result["status"] == "success":
            generated_thumbnail_url = thumb_result["image_url"]
            print(f"[LANGGRAPH][NODE][OK] image_generator: Thumbnail → {generated_thumbnail_url[:60]}...")
        else:
            print(f"[LANGGRAPH][NODE][WARN] image_generator: Thumbnail FAILED — {thumb_result.get('error', 'unknown')}")
    else:
        print("[LANGGRAPH][NODE][INFO] image_generator: No thumbnail_prompt — skipping thumbnail generation.")

    # Build output
    success_count = sum(1 for u in generated_urls if u.status == "success")
    fail_count = sum(1 for u in generated_urls if u.status == "failed")
    urls_json = json.dumps([u.model_dump() for u in generated_urls])

    thumb_status = "generated" if generated_thumbnail_url else "skipped"
    print(f"[LANGGRAPH][NODE][OK] image_generator complete: "
          f"{success_count} success, {fail_count} failed out of {total} images | thumbnail: {thumb_status}")

    return {
        "generated_image_urls": urls_json,
        "generated_thumbnail_url": generated_thumbnail_url
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: LANGGRAPH PIPELINE ASSEMBLY
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline() -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Pipeline flow:
        START → rewrite_title → rewrite_description → rewrite_hashtags
              → transcript_steps_maker → script_writer → script_polish
              → image_allocator → image_placer → thumbnail_analyzer
              → image_generator → END
    """
    print("[LANGGRAPH] Building graph...")
    builder = StateGraph(MetadataState)

    # Register all nodes
    builder.add_node("rewrite_title",            rewrite_title_node)
    builder.add_node("rewrite_description",      rewrite_description_node)
    builder.add_node("rewrite_hashtags",         rewrite_hashtags_node)
    builder.add_node("transcript_steps_maker",   transcript_steps_maker_node)
    builder.add_node("script_writer",            script_writer_node)
    builder.add_node("script_polish",            script_polish_node)
    builder.add_node("image_allocator",          image_allocator_node)
    builder.add_node("image_placer",             image_placer_node)
    builder.add_node("thumbnail_analyzer",       thumbnail_analyzer_node)
    builder.add_node("image_generator",          image_generator_node)

    # Wire the edges (sequential pipeline)
    builder.add_edge(START,                      "rewrite_title")
    builder.add_edge("rewrite_title",            "rewrite_description")
    builder.add_edge("rewrite_description",      "rewrite_hashtags")
    builder.add_edge("rewrite_hashtags",         "transcript_steps_maker")
    builder.add_edge("transcript_steps_maker",   "script_writer")
    builder.add_edge("script_writer",            "script_polish")
    builder.add_edge("script_polish",            "image_allocator")
    builder.add_edge("image_allocator",          "image_placer")
    builder.add_edge("image_placer",             "thumbnail_analyzer")
    builder.add_edge("thumbnail_analyzer",       "image_generator")
    builder.add_edge("image_generator",          END)

    graph = builder.compile()
    print("[LANGGRAPH][OK] Pipeline compiled successfully.")
    return graph


# Singleton compiled graph — initialized once at import time
pipeline = build_pipeline()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: PIPELINE ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    title: str,
    description: str,
    hashtags: str,
    transcript: str,
    language: str,
    min_script_word_count: int,
    default_image_count: int,
    thumbnail_url: Optional[str] = None,
) -> dict:
    """Run the LangGraph pipeline and return deserialized results.

    All per-request configuration (language, word count, image count)
    is passed through the state so concurrent requests don't overwrite each other.
    """
    print(f"[REWRITE_METADATA] Running LangGraph pipeline (language={language}, "
          f"min_words={min_script_word_count}, images={default_image_count})...")

    result = pipeline.invoke({
        # ── Per-request configuration ──
        "language":                  language,
        "min_script_word_count":     min_script_word_count,
        "default_image_count":       default_image_count,
        # ── Original fetched data ──
        "title":                     title,
        "description":               description,
        "hashtags":                  hashtags,
        "transcript":                transcript,
        "thumbnail_url":             thumbnail_url,
        # ── Pipeline outputs (initialized to None) ──
        "rewritten_title":           None,
        "rewritten_description":     None,
        "rewritten_hashtags":        None,
        "script_steps":              None,
        "written_steps_draft":       None,
        "written_steps_list":        None,
        "final_script":              None,
        "image_allocations":         None,
        "image_plan":                None,
        "annotated_transcript":      None,
        "generated_image_urls":      None,
        "thumbnail_prompt":          None,
        "generated_thumbnail_url":   None,
    })
    print("[REWRITE_METADATA][OK] Pipeline complete.")

    # Deserialize JSON strings → Pydantic objects
    script_steps_obj = None
    raw_steps = result.get("script_steps")
    if raw_steps:
        try:
            script_steps_obj = TranscriptStepsOutput.model_validate_json(raw_steps)
        except Exception as e:
            print(f"[REWRITE_METADATA][WARN] Could not parse script_steps: {e}")

    image_plan_obj = None
    raw_image_plan = result.get("image_plan")
    if raw_image_plan:
        try:
            image_plan_obj = ImagePlanOutput.model_validate_json(raw_image_plan)
        except Exception as e:
            print(f"[REWRITE_METADATA][WARN] Could not parse image_plan: {e}")

    generated_image_urls_obj = None
    raw_urls = result.get("generated_image_urls")
    if raw_urls:
        try:
            generated_image_urls_obj = [GeneratedImageURL(**u) for u in json.loads(raw_urls)]
        except Exception as e:
            print(f"[REWRITE_METADATA][WARN] Could not parse generated_image_urls: {e}")

    return {
        "rewritten_title":         result["rewritten_title"],
        "rewritten_description":   result["rewritten_description"],
        "rewritten_hashtags":      result["rewritten_hashtags"],
        "script_steps":            script_steps_obj,
        "final_script":            result.get("final_script"),
        "image_plan":              image_plan_obj,
        "annotated_transcript":    result.get("annotated_transcript"),
        "generated_image_urls":    generated_image_urls_obj,
        "generated_thumbnail_url": result.get("generated_thumbnail_url"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="YouTube Data Fetcher + Rewriter",
    version="8.0.0",
    description="FastAPI + LangGraph pipeline for YouTube video content regeneration "
                "with multi-language support.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.post("/fetch-video-data", response_model=VideoDataResponse)
def fetch_video_data_endpoint(request: YouTubeURLRequest):
    """Fetch YouTube data and run the full LangGraph rewriting + image pipeline.

    Accepts: url (YouTube URL), language (target language for all content),
    min_script_word_count (minimum words for the script),
    default_image_count (target number of images to generate).
    """
    try:
        url = request.url
        language = request.language
        min_script_word_count = request.min_script_word_count
        default_image_count = request.default_image_count

        print("\n" + "=" * 60)
        print(f"[REQUEST] URL: {url}")
        print(f"[REQUEST] Language: {language}")
        print(f"[REQUEST] Min script word count: {min_script_word_count}")
        print(f"[REQUEST] Default image count: {default_image_count}")
        print("=" * 60)

        # Step 1: Fetch raw data
        video_id = extract_video_id(url)
        if not video_id:
            raise HTTPException(status_code=400, detail="Invalid YouTube URL.")

        metadata = fetch_metadata(video_id)
        if not metadata:
            raise HTTPException(status_code=404, detail="Video not found or metadata inaccessible.")

        hashtags = build_metadata_string(metadata)
        transcript = fetch_transcript(video_id, language=language)
        if not transcript:
            raise HTTPException(status_code=404, detail="No transcript found for this video.")

        thumbnail_url = fetch_thumbnail_url(url)
        print("[STEP 1][OK] Raw data fetched.")

        # Step 2: Run LangGraph pipeline
        print("\n[STEP 2] Running LangGraph pipeline...")
        rewritten = run_pipeline(
            title=metadata["title"],
            description=metadata["description"],
            hashtags=hashtags,
            transcript=transcript,
            language=language,
            min_script_word_count=min_script_word_count,
            default_image_count=default_image_count,
            thumbnail_url=thumbnail_url
        )
        print("[STEP 2][OK] Pipeline complete.")

        print("\n[REQUEST][OK] Returning full response.")
        print("=" * 60 + "\n")

        return VideoDataResponse(
            video_id=video_id,
            title=metadata["title"],
            description=metadata["description"],
            tags=metadata["tags"],
            hashtags=hashtags,
            transcript=transcript,
            thumbnail_url=thumbnail_url,
            rewritten_title=rewritten["rewritten_title"],
            rewritten_description=rewritten["rewritten_description"],
            rewritten_hashtags=rewritten["rewritten_hashtags"],
            script_steps=rewritten["script_steps"],
            final_script=rewritten["final_script"],
            image_plan=rewritten["image_plan"],
            annotated_transcript=rewritten["annotated_transcript"],
            generated_image_urls=rewritten["generated_image_urls"],
            generated_thumbnail_url=rewritten["generated_thumbnail_url"]
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REQUEST][ERROR] {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ── Direct run for quick testing ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("[STARTUP] Visit: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
