import os
from dotenv import load_dotenv
load_dotenv()

try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    POLICY_DATA_PATH = os.getenv("POLICY_DATA_PATH")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")
    
    # Cast strings to integers/floats safely with fallbacks
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    TOP_K = int(os.getenv("TOP_K", 4))
    
    # Mirror back to os.environ if your underlying libraries require it
    os.environ['GROQ_API_KEY'] = GROQ_API_KEY or ""
    os.environ['GROQ_MODEL'] = GROQ_MODEL or ""
except Exception as e:
    print(f"Configuration Error: {e}")
        
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not set. The LLM will not be able to generate responses.")
