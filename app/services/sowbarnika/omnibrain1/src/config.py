import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Retrieve API keys and settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_BASE_URL = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST")

# Determine if the environment should run in mock/offline mode
OMNIBRAIN_ENV = os.getenv("OMNIBRAIN_ENV", "production").lower()

# Fallback to mock mode automatically if main OpenAI API key is missing or placeholder
has_no_openai = not OPENAI_API_KEY or "your-" in OPENAI_API_KEY or "your_" in OPENAI_API_KEY

if OMNIBRAIN_ENV == "mock" or has_no_openai:
    IS_MOCK = True
else:
    IS_MOCK = False

# Database path
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "historical_stock_data.db"))
