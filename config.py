import os
from dotenv import load_dotenv

load_dotenv()

# --- JWT Configuration ---
# Fixed: Using JWT_SECRET_KEY from .env instead of SECRET_KEY
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "fallback_secret_key_change_in_production")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# --- MongoDB Configuration ---
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "videoanalysis")

# --- Application Configuration ---
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))
PROCESS_TIMEOUT_SECONDS = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "900"))

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- CORS Configuration ---
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["http://localhost:3000"]

# --- Super Admin Configuration ---
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME", "admin")
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "")
SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL", "admin@example.com")

# --- File Storage Configuration ---
BULK_RESULTS_BASE_DIR = os.getenv("BULK_RESULTS_BASE_DIR", "bulk_analysis_reports")