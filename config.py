from agents import (

    AsyncOpenAI,
    OpenAIChatCompletionsModel,
  
)
import os
from dotenv import load_dotenv

load_dotenv()



# --- Load API keys from .env ---
first_api_key = os.getenv("first")
second_api_key = os.getenv("second")
third_api_key = os.getenv("third")
fourth_api_key = os.getenv("fourth")
fifth_api_key = os.getenv("fifth")

# -----------------------------
# 1️⃣ First client and model
# -----------------------------
first_client = AsyncOpenAI(
    api_key=first_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

first_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=first_client
)

# -----------------------------
# 2️⃣ Second client and model
# -----------------------------
second_client = AsyncOpenAI(
    api_key=second_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

second_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=second_client
)

# -----------------------------
# 3️⃣ Third client and model
# -----------------------------
third_client = AsyncOpenAI(
    api_key=third_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

third_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=third_client
)

# -----------------------------
# 4️⃣ Fourth client and model
# -----------------------------
fourth_client = AsyncOpenAI(
    api_key=fourth_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

fourth_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=fourth_client
)

# -----------------------------
# 5️⃣ Fifth client and model
# -----------------------------
fifth_client = AsyncOpenAI(
    api_key=fifth_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

fifth_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=fifth_client
)

# -----------------------------
# ✅ Now you have first_model, second_model, third_model, fourth_model, fifth_model
# -----------------------------


