import os

# Load API keys from .env file if it exists in the current directory or home directory
for env_path in [".env", os.path.expanduser("~/.env")]:
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip quotes if present
                    val = val.strip().strip("'").strip('"')
                    os.environ[key.strip()] = val

# Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Active provider: exclusively 'gemini'
DEFAULT_PROVIDER = "gemini"

# Model configurations
MODELS = {
    "openai": {
        "vlm": "gpt-4o",
        "llm": "gpt-4o-mini",
        "embed": "text-embedding-3-small"
    },
    "gemini": {
        "vlm": "gemini-2.5-flash",
        "llm": "gemini-2.5-flash",
        "embed": "text-embedding-004"
    }
}

# General configurations
MAX_SELF_RAG_RETRIES = 3

def get_vlm_model(provider=None):
    prov = provider or DEFAULT_PROVIDER
    return MODELS[prov]["vlm"]

def get_llm_model(provider=None):
    prov = provider or DEFAULT_PROVIDER
    return MODELS[prov]["llm"]

def get_embed_model(provider=None):
    prov = provider or DEFAULT_PROVIDER
    return MODELS[prov]["embed"]

def get_api_key(provider=None):
    prov = provider or DEFAULT_PROVIDER
    if prov == "gemini":
        return GEMINI_API_KEY
    return OPENAI_API_KEY

def print_status():
    print(f"--- OmniBrain Configuration ---")
    print(f"Default Provider: {DEFAULT_PROVIDER.upper()}")
    print(f"Gemini API Key Configured: {'Yes' if GEMINI_API_KEY else 'No (set GEMINI_API_KEY environment variable)'}")
    print(f"OpenAI API Key Configured: {'Yes' if OPENAI_API_KEY else 'No (set OPENAI_API_KEY environment variable)'}")
    print(f"VLM Model: {get_vlm_model()}")
    print(f"LLM Model: {get_llm_model()}")
    print(f"Embed Model: {get_embed_model()}")
    print(f"---------------------------------")
