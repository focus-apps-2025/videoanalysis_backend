# main.py - Comprehensive Backend with RBAC, Dashboards, and Analysis Features

import os
import sys
import io as _io
import logging
import contextlib
import hashlib
import zipfile
import shutil
import tempfile
import json # For json.dump in structured download, also for clean_results logic
import re # Needed for _sanitize_path_segment and other regex ops

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime as dt, timedelta,datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager

import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Request, Form, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, JSONResponse, FileResponse, Response
from pydantic import BaseModel, Field # Added Field for MongoDB _id alias

# --- RBAC Specific Imports ---
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # For login endpoint

# --- MongoDB Async Client ---
from motor.motor_asyncio import AsyncIOMotorClient # For async MongoDB operations

from utils import verify_password, get_password_hash



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Dude import UnifiedMediaAnalyzer # Assuming HELLO.py is in the same directory
import uuid


analysis_tasks_collection = None 

load_dotenv()


# -----------------------------
# Logging Configuration
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("citnow_analyzer")

# -----------------------------
# Application Configuration (from .env)
# -----------------------------
APP_TITLE = "CitNow Analyzer API"
APP_VERSION = "1.0.0"

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "2"))
PROCESS_TIMEOUT_SECONDS = int(os.getenv("PROCESS_TIMEOUT_SECONDS", "900")) # 15 minutes

BULK_RESULTS_BASE_DIR = os.getenv("BULK_RESULTS_BASE_DIR", "bulk_analysis_reports")
os.makedirs(BULK_RESULTS_BASE_DIR, exist_ok=True)
logger.info(f"Bulk analysis reports will be stored in: {BULK_RESULTS_BASE_DIR}")
# CORS Origins
_frontend_urls_env = os.getenv(
    "FRONTEND_URLS",
    "https://focusadmin.focusengineeringapp.com,https://focus-user.focusengineeringapp.com,https://videoapi.focusengineeringapp.com"
)
CORS_ORIGINS = [url.strip() for url in _frontend_urls_env.split(',')]


# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-super-secret-key-change-this-in-prod!") # IMPORTANT: Change this in production
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440")) # 24 hours (for simpler token renewal on frontend)

# Super Admin Defaults (for initial setup in development/first run)
SUPER_ADMIN_USERNAME = os.getenv("SUPER_ADMIN_USERNAME", "admin")
SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "adminpass") # IMPORTANT: Change this in production
SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL", "admin@example.com")


# -----------------------------
# Globals for Analyzer and ThreadPool
# -----------------------------
analyzer: Optional[UnifiedMediaAnalyzer] = None
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT) # To limit concurrent CPU-bound analysis tasks
batch_cancellation_flags = {}  # Track cancellation requests for background tasks

# -----------------------------
# Enums (Moved from existing main.py structure)
# -----------------------------
class BatchStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STOPPING = "stopping"

# -----------------------------
# Pydantic Models (SIMPLIFIED - No Dealers)
# -----------------------------

# --- JWT Models ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None
    role: Optional[str] = None
    dealer_id: Optional[str] = None

# --- User Models - SIMPLIFIED ---
class UserBase(BaseModel):
    username: str
    email: Optional[str] = None
    showroom_name: Optional[str] = None

class UserCreate(UserBase):
    password: str
    role: str = "dealer_user" # Default role for new users
    dealer_id: Optional[str] = None # Simple string field for dealer identification
class UserUpdate(BaseModel):
    # every field optional – update only what is provided
    username: Optional[str] = None
    email:  Optional[str] = None
    role:   Optional[str] = None        # 'dealer_admin' | 'super_admin'
    password: Optional[str] = None      # will be re-hashed if present
    dealer_id: Optional[str] = None     # simple string (your simplified schema)
    showroom_name: Optional[str] = None

class UserInDB(UserBase):
    id: str = Field(alias="_id")  # This should accept ObjectId converted to string
    hashed_password: str
    role: str
    dealer_id: Optional[str] = None
    created_at: dt
    updated_at: dt

    class Config:
        populate_by_name = True  # Allow both alias and field name
        json_encoders = {ObjectId: str}  # Convert ObjectId to string in JSON

# --- Analysis Request/Response Models ---
class AnalysisRequest(BaseModel):
    citnow_url: str
    transcription_language: str = "auto"
    target_language: str = "en"

class AnalysisResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    result_id: Optional[str] = None
    task_id: Optional[str] = None  
    results: Optional[Dict[str, Any]] = None

class BatchCreateResponse(BaseModel):
    success: bool
    batch_id: str
    total_urls: int
    message: str
    status: Optional[str] = None
    processed_urls: Optional[int] = None
    failed_urls: Optional[int] = None
    created_at: Optional[dt] = None
    updated_at: Optional[dt] = None
    filename: Optional[str] = None
    submitted_by_user_id: Optional[str] = None
    dealer_id: Optional[str] = None

class BatchStatusResponse(BaseModel):
    batch_id: str
    status: str
    total_urls: int
    processed_urls: int
    failed_urls: int
    progress_percentage: float
    current_url: Optional[str] = None
    can_cancel: bool = False

# --- Dashboard Models - SIMPLIFIED ---
class DealerSummary(BaseModel):
    dealer_id: str
    total_videos: int
    avg_overall_quality: float

class SuperAdminDashboardOverview(BaseModel):
    total_videos_analyzed: int
    average_overall_quality: float
    quality_distribution: Dict[str, int]
    dealers_summary: List[DealerSummary]
    last_updated: dt

class RecentAnalysis(BaseModel):
    id: str = Field(alias="_id")
    original_url: str = Field(alias="input_source")
    overall_quality_label: Optional[str]
    overall_quality_score: Optional[float]
    created_at: dt
    status: Optional[str] = "completed"
    error_message: Optional[str] = None

class DealerAdminDashboardOverview(BaseModel):
    dealer_id: str
    total_videos_analyzed: int
    average_overall_quality: float
    quality_distribution: Dict[str, int]
    low_quality_video_count: int
    low_quality_audio_count: int
    recent_analyses: List[RecentAnalysis]
    last_updated: dt

class ProfileUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    current_password: Optional[str] = None
    new_password: Optional[str] = None



# -----------------------------
# MongoDB Configuration (SIMPLIFIED - No Dealers Collection)
# -----------------------------
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "citnow_analyzer")

# Global variables for MongoDB client and collections
client: AsyncIOMotorClient = None
db = None
results_collection = None
batch_collection = None
excel_data_collection = None
users_collection = None

async def connect_to_mongo():
    """Establishes MongoDB connection and assigns collections to global variables."""
    global client, db, results_collection, batch_collection, excel_data_collection, users_collection, analysis_tasks_collection  # ADD analysis_tasks_collection
    try:
        client = AsyncIOMotorClient(MONGODB_URI)
        db = client[MONGODB_DB_NAME]
        results_collection = db["analysis_results"]
        batch_collection = db["batch_jobs"]
        excel_data_collection = db["excel_upload_data"]
        users_collection = db["users"]
        analysis_tasks_collection = db["analysis_tasks"]  # ADD THIS LINE
        # REMOVED: dealers_collection
        logger.info("MongoDB connection established.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)

async def close_mongo_connection():
    """Closes the MongoDB connection."""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed.")

async def create_mongo_indexes():
    """Creates necessary indexes for collections to optimize queries."""
    if db is None:
        logger.error("Database not connected. Cannot create indexes.")
        return

    try:
        # Indexes for analysis_results collection
        await results_collection.create_index([("batch_id", 1)])
        await results_collection.create_index([("created_at", -1)])
        await results_collection.create_index([("dealer_id", 1)]) # Keep for filtering
        await results_collection.create_index([("submitted_by_user_id", 1)])
        await results_collection.create_index([("overall_quality_label", 1)])
        await results_collection.create_index([("video_quality_label", 1)])
        await results_collection.create_index([("audio_clarity_level", 1)])
        await results_collection.create_index([("status", 1)])

        # Indexes for batch_jobs collection
        await batch_collection.create_index([("status", 1)])
        await batch_collection.create_index([("created_at", -1)])
        await batch_collection.create_index([("dealer_id", 1)]) # Keep for filtering
        await batch_collection.create_index([("submitted_by_user_id", 1)])

        # Indexes for users collection
        await users_collection.create_index([("username", 1)], unique=True)
        await users_collection.create_index([("dealer_id", 1)]) # Keep for quick lookups
        
        # NEW: Indexes for analysis_tasks collection
        await analysis_tasks_collection.create_index([("task_id", 1)], unique=True)
        await analysis_tasks_collection.create_index([("submitted_by_user_id", 1)])
        await analysis_tasks_collection.create_index([("dealer_id", 1)])
        await analysis_tasks_collection.create_index([("status", 1)])
        await analysis_tasks_collection.create_index([("created_at", -1)])
        await analysis_tasks_collection.create_index([("expires_at", 1)], expireAfterSeconds=0)
        
        logger.info("MongoDB indexes created/ensured.")
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {e}")



# -----------------------------
# RBAC - JWT Token Utilities
# -----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Creates a JWT access token with a configurable expiration time."""
    to_encode = data.copy()
    if expires_delta:
        expire = dt.utcnow() + expires_delta
    else:
        expire = dt.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt

# -----------------------------
# RBAC - FastAPI Dependency Functions for Authentication & Authorization
# -----------------------------

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """
    Dependency function that decodes a JWT, validates it, and fetches the
    corresponding user from the database.
    """
    if users_collection is None:
        logger.error("Database users_collection not initialized during get_current_user call.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Server error: Database not initialized.")

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        username: str = payload.get("sub")
        user_id: str = payload.get("user_id")
        role: str = payload.get("role")
        dealer_id_str: Optional[str] = payload.get("dealer_id")
        if username is None or user_id is None or role is None:
            logger.warning("Token payload missing essential fields: sub, user_id, or role.")
            raise credentials_exception
        token_data = TokenData(username=username, user_id=user_id, role=role, dealer_id=dealer_id_str)
    except JWTError:
        logger.warning("JWT decoding failed or token is invalid.")
        raise credentials_exception

    user_doc = await users_collection.find_one({"_id": ObjectId(token_data.user_id)})
    if user_doc is None:
        logger.warning(f"User with ID {token_data.user_id} from token not found in DB.")
        raise credentials_exception

    # Convert ObjectId fields to strings
    user_doc["_id"] = str(user_doc["_id"])
    # dealer_id is now stored as string, no conversion needed
    
    return UserInDB(**user_doc)

async def get_current_super_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Ensures the current authenticated user has the 'super_admin' role."""
    if current_user.role != "super_admin":
        logger.warning(f"User {current_user.username} attempted unauthorized Super Admin access.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized: Super Admin role required.")
    return current_user

async def get_current_dealer_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Ensures the current authenticated user has the 'dealer_admin' role."""
    if current_user.role != "dealer_admin":
        logger.warning(f"User {current_user.username} attempted unauthorized Dealer Admin access.")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized: Dealer Admin role required.")
    return current_user

# -----------------------------
# Initial Super Admin Creation
# -----------------------------
async def create_initial_super_admin_if_not_exists():
    """Creates a default Super Admin user if one does not already exist."""
    if users_collection is None:
        logger.error("MongoDB users collection not initialized. Cannot create super admin.")
        return

    if not SUPER_ADMIN_USERNAME or not SUPER_ADMIN_PASSWORD:
        logger.warning("SUPER_ADMIN_USERNAME or SUPER_ADMIN_PASSWORD not set in config. Skipping initial super admin creation.")
        return

    existing_admin = await users_collection.find_one({"username": SUPER_ADMIN_USERNAME})
    if not existing_admin:
        hashed_password = get_password_hash(SUPER_ADMIN_PASSWORD)
        admin_user = {
            "username": SUPER_ADMIN_USERNAME,
            "email": SUPER_ADMIN_EMAIL,
            "hashed_password": hashed_password,
            "role": "super_admin",
            "dealer_id": None,
            "created_at": dt.utcnow(),
            "updated_at": dt.utcnow()
        }
        await users_collection.insert_one(admin_user)
        logger.info(f"Created initial Super Admin: '{SUPER_ADMIN_USERNAME}'")
    else:
        logger.info(f"Super Admin '{SUPER_ADMIN_USERNAME}' already exists.")

# -----------------------------
# Utility Functions
# -----------------------------
import numpy as np

def clean_results(obj):
    """Recursively cleans analysis results to be JSON serializable."""
    if isinstance(obj, dict):
        return {key: clean_results(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_results(item) for item in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dt):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                          np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.str_):
        return str(obj)
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        logger.debug(f"clean_results encountered unhandled type {type(obj)}. Converting to str.")
        return str(obj)

def _sanitize_path_segment(name: str) -> str:
    """Sanitizes a string to be a safe filename or directory name."""
    if not name:
        return "unknown_segment"
    safe_name = re.sub(r'[^\w\-\.]', '_', name)
    return safe_name.strip('_')[:100]

# -----------------------------
# FastAPI Lifespan Events
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    logger.info("Application starting up...")

    # 1. Connect to MongoDB and create indexes
    await connect_to_mongo()
    await create_mongo_indexes()
    
    # 2. Create initial Super Admin (if not exists)
    await create_initial_super_admin_if_not_exists()

    # 3. Initialize and preload UnifiedMediaAnalyzer models
    logger.info("Initializing UnifiedMediaAnalyzer instance and pre-loading models...")
    analyzer = UnifiedMediaAnalyzer()
    try:
        analyzer.load_pretrained_models()
        logger.info("Pre-loaded essential models for UnifiedMediaAnalyzer.")
    except Exception:
        logger.exception("Could not pre-load all models for UnifiedMediaAnalyzer (continuing startup).")

    # 4. Start background cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    # 5. Shutdown cleanup
    logger.info("Application shutting down...")
    batch_cancellation_flags.clear()
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    logger.info("Shutting down ThreadPoolExecutor.")
    try:
        executor.shutdown(wait=True, cancel_futures=True)
    except Exception:
        logger.exception("Error during ThreadPoolExecutor shutdown.")
    
    logger.info("Closing MongoDB connection.")
    await close_mongo_connection()

# -----------------------------
# CORS Middleware - COMPLETE FIX
# -----------------------------

# Add this RIGHT AFTER creating your FastAPI app

# Add this RIGHT AFTER creating your FastAPI app
app = FastAPI(title=APP_TITLE, version=APP_VERSION, lifespan=lifespan, default_response_class=ORJSONResponse)


# COMPLETE CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # dynamically loaded from env
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{rest_of_path:path}")
async def preflight_handler():
    return JSONResponse(content={"status": "ok"})

# Add this middleware to handle CORS headers
@app.middleware("http")
async def add_cors_header(request: Request, call_next):
    response = await call_next(request)
    origin = request.headers.get("origin")
    
    allowed_origins = CORS_ORIGINS  # use same allowed list
    
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, Accept"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    
    return response
#-----------------------------
# Authentication Endpoints
# -----------------------------
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_doc = await users_collection.find_one({"username": form_data.username})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # dealer_id is now stored as string, no conversion needed
    dealer_id_str = user_doc.get("dealer_id")
    showroom_name = user_doc.get("showroom_name")

    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_doc["username"],
            "user_id": str(user_doc["_id"]),
            "role": user_doc["role"],
            "dealer_id": dealer_id_str,
	    "showroom_name": showroom_name,
        },
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}


# -----------------------------
# Separate Authentication Endpoints for Different Portals
# -----------------------------

@app.post("/admin/login", response_model=Token)
async def admin_portal_login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint for Admin Portal (Super Admin + Dealer Admin only)
    """
    user_doc = await users_collection.find_one({"username": form_data.username})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # RESTRICT: Only allow admin roles to login via admin portal
    if user_doc["role"] not in ["super_admin", "dealer_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please use the dealer portal to login"
        )
    
    dealer_id_str = user_doc.get("dealer_id")

    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_doc["username"],
            "user_id": str(user_doc["_id"]),
            "role": user_doc["role"],
            "dealer_id": dealer_id_str,
        },
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/dealer/login", response_model=Token)
async def dealer_portal_login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Login endpoint for Dealer User Portal (Dealer Users only)
    """
    user_doc = await users_collection.find_one({"username": form_data.username})
    if not user_doc or not verify_password(form_data.password, user_doc["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # RESTRICT: Only allow dealer users to login via dealer portal
    if user_doc["role"] != "dealer_user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please use the admin portal to login"
        )
    dealer_id_str = user_doc.get("dealer_id")
    dealer_admin = await users_collection.find_one({
        "dealer_id": dealer_id_str, 
        "role": "dealer_admin"
    })
    dealership_showroom_name = dealer_admin.get("showroom_name") if dealer_admin else None

    access_token_expires = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user_doc["username"],
            "user_id": str(user_doc["_id"]),
            "role": user_doc["role"],
            "dealer_id": dealer_id_str,
            "showroom_name": dealership_showroom_name,  
        },
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}
# Dealer User Specific Endpoints
# -----------------------------

@app.get("/dealer/dashboard/overview")
async def get_dealer_user_dashboard(current_user: UserInDB = Depends(get_current_user)):
    """
    Simplified dashboard for dealer users (all dealership data)
    """
    if current_user.role != "dealer_user":
        raise HTTPException(status_code=403, detail="Dealer users only")
    
    if not current_user.dealer_id:
        raise HTTPException(status_code=400, detail="User has no assigned dealer_id")
    
    # Get ALL dealership stats (not just user's own)
    dealer_videos_count = await results_collection.count_documents({
        "dealer_id": current_user.dealer_id,
        "status": BatchStatus.COMPLETED
    })
    
    # Recent analyses from entire dealership
    recent_analyses = await results_collection.find({
        "dealer_id": current_user.dealer_id,
        "status": BatchStatus.COMPLETED
    }).sort("created_at", -1).limit(10).to_list(10)
    
    # User's personal stats
    user_videos_count = await results_collection.count_documents({
        "submitted_by_user_id": str(current_user.id),
        "status": BatchStatus.COMPLETED
    })
    
    return {
        "dealer_videos_analyzed": dealer_videos_count,
        "user_videos_analyzed": user_videos_count,
        "recent_analyses": [clean_results(r) for r in recent_analyses],
        "dealer_id": current_user.dealer_id
    }

@app.get("/dealer/results")
async def get_dealer_results(
    limit: int = 50,
    current_user: UserInDB = Depends(get_current_user)
):
    """Dealer user can see ALL results from their dealership"""
    if current_user.role != "dealer_user":
        raise HTTPException(status_code=403, detail="Dealer users only")
    
    if not current_user.dealer_id:
        raise HTTPException(status_code=400, detail="User has no assigned dealer_id")
    
    results = await results_collection.find({
        "dealer_id": current_user.dealer_id,
        "status": BatchStatus.COMPLETED
    }).sort("created_at", -1).limit(min(limit, 100)).to_list(None)
    
    return [clean_results(r) for r in results]

@app.get("/dealer/my-results")
async def get_my_personal_results(
    limit: int = 50,
    current_user: UserInDB = Depends(get_current_user)
):
    """Dealer user can see ONLY their own results"""
    if current_user.role != "dealer_user":
        raise HTTPException(status_code=403, detail="Dealer users only")
    
    results = await results_collection.find({
        "submitted_by_user_id": str(current_user.id),
        "status": BatchStatus.COMPLETED
    }).sort("created_at", -1).limit(min(limit, 100)).to_list(None)
    
    return [clean_results(r) for r in results]

@app.delete("/results/{result_id}")
async def delete_result(result_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(result_id):
        raise HTTPException(status_code=400, detail="Invalid Result ID format.")

    result = await results_collection.find_one({"_id": ObjectId(result_id)})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")

    # Authorization checks
    if current_user.role == "dealer_admin":
        # Dealer admin can only delete results from their dealer
        if current_user.dealer_id != result.get("dealer_id"):
            raise HTTPException(status_code=403, detail="Not authorized to delete this result.")
    
    elif current_user.role == "dealer_user":
        # Dealer user can only delete their OWN results
        if result.get("submitted_by_user_id") != str(current_user.id):
            raise HTTPException(status_code=403, detail="You can only delete your own analysis results.")
    
    elif current_user.role != "super_admin":
        raise HTTPException(status_code=403, detail="Not authorized to delete results.")

    delete_operation = await results_collection.delete_one({"_id": ObjectId(result_id)})
    if delete_operation.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Result not found for deletion (may have been already deleted).")

    return JSONResponse(status_code=200, content={"success": True, "message": "Result deleted successfully."})
@app.get("/dealer/my-analysis-tasks")
async def get_my_dealer_analysis_tasks(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    """Dealer user can only see their own analysis tasks"""
    if current_user.role != "dealer_user":
        raise HTTPException(status_code=403, detail="Dealer users only")
        
    if analysis_tasks_collection is None:
        raise HTTPException(status_code=500, detail="Analysis tasks collection not initialized")
        
    tasks_cursor = analysis_tasks_collection.find({
        "submitted_by_user_id": str(current_user.id)
    }).sort("created_at", -1).limit(limit)
    
    tasks = await tasks_cursor.to_list(length=limit)
    
    for task in tasks:
        if '_id' in task:
            task.pop('_id')
    
    return {"tasks": tasks}


@app.get("/users/me", response_model=UserInDB)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    return current_user

@app.put("/users/me", response_model=UserInDB)
async def update_user_profile(
    profile_data: ProfileUpdate,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Update current user's profile (username, email, password)
    """
    updates = {}
    
    # Check if username is being updated
    if profile_data.username is not None:
        if profile_data.username.strip() == "":
            raise HTTPException(status_code=400, detail="Username cannot be empty")
        
        # Check if username already exists (excluding current user)
        existing_user = await users_collection.find_one({
            "username": profile_data.username,
            "_id": {"$ne": ObjectId(current_user.id)}
        })
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        updates["username"] = profile_data.username.strip()
    
    # Check if email is being updated
    if profile_data.email is not None:
        if profile_data.email.strip() == "":
            raise HTTPException(status_code=400, detail="Email cannot be empty")
        
        # Basic email validation
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, profile_data.email):
            raise HTTPException(status_code=400, detail="Invalid email format")
        
        # Check if email already exists (excluding current user)
        existing_user = await users_collection.find_one({
            "email": profile_data.email,
            "_id": {"$ne": ObjectId(current_user.id)}
        })
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already exists")
        updates["email"] = profile_data.email.strip()
    
    # Handle password change if requested - NO CURRENT PASSWORD REQUIRED
    if profile_data.new_password is not None:
        # Validate new password
        if len(profile_data.new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters long")
        
        # Hash and set new password
        updates["hashed_password"] = get_password_hash(profile_data.new_password)
    
    # If no valid updates, return error
    if not updates:
        raise HTTPException(status_code=400, detail="No valid fields to update")
    
    # Add update timestamp
    updates["updated_at"] = dt.utcnow()
    
    # Update user in database
    result = await users_collection.update_one(
        {"_id": ObjectId(current_user.id)},
        {"$set": updates}
    )
    
    if result.modified_count == 0:
        raise HTTPException(status_code=500, detail="Failed to update profile")
    
    # Fetch and return updated user
    updated_user_doc = await users_collection.find_one({"_id": ObjectId(current_user.id)})
    if not updated_user_doc:
        raise HTTPException(status_code=404, detail="User not found after update")
    
    # Convert ObjectId to string
    updated_user_doc["_id"] = str(updated_user_doc["_id"])
    
    return UserInDB(**updated_user_doc)


# ===============================
# User Management Endpoints (RBAC Enhanced)
# ===============================

@app.get("/users/", response_model=List[UserInDB])
async def read_users(current_user: UserInDB = Depends(get_current_user)):
    """
    Super Admin → all users
    Dealer Admin → users belonging to their own dealer_id
    Dealer User → forbidden
    """
    if current_user.role == "super_admin":
        users_cursor = users_collection.find()
    elif current_user.role == "dealer_admin":
        if not current_user.dealer_id:
            raise HTTPException(403, detail="Dealer Admin has no assigned dealer_id.")
        users_cursor = users_collection.find({"dealer_id": current_user.dealer_id})
    else:
        raise HTTPException(403, detail="Not authorized to view users.")

    users_list = await users_cursor.to_list(None)

    for user_doc in users_list:
        user_doc["_id"] = str(user_doc["_id"])
    return [UserInDB(**u) for u in users_list]


@app.post("/users/", response_model=UserInDB, status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate, current_user: UserInDB = Depends(get_current_user)):
    """
    Super Admin → can create any user
    Dealer Admin → can only create dealer_user under same dealer_id
    """
    existing_user = await users_collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered.")

    # Determine allowed creation scope
    if current_user.role == "super_admin":
        allowed_role = user.role  # super_admin can create any role
        allowed_dealer = user.dealer_id
        allowed_showroom = user.showroom_name
    elif current_user.role == "dealer_admin":
        # Dealer admin can only create dealer_user (not dealer_admin or super_admin)
        if user.role not in ["dealer_user", "dealer_admin"]:
            raise HTTPException(status_code=403, detail="Dealer admins can only create dealer_user accounts")
        allowed_role = user.role
        allowed_dealer = current_user.dealer_id  # Force their dealer_id
        allowed_showroom = current_user.showroom_name

    else:
        raise HTTPException(403, detail="Not authorized to create user.")

    hashed_password = get_password_hash(user.password)
    user_doc = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed_password,
        "role": allowed_role,
        "dealer_id": allowed_dealer,
	"showroom_name": allowed_showroom,
        "created_at": dt.utcnow(),
        "updated_at": dt.utcnow()
    }

    inserted = await users_collection.insert_one(user_doc)
    user_doc["_id"] = str(inserted.inserted_id)
    return UserInDB(**user_doc)


@app.put("/users/{user_id}", response_model=UserInDB)
async def update_user(user_id: str, payload: UserUpdate, current_user: UserInDB = Depends(get_current_user)):
    """
    Super Admin → can edit any user
    Dealer Admin → can edit only users under their own dealer_id
    """
    if not ObjectId.is_valid(user_id):
        raise HTTPException(400, detail="Invalid user_id")

    user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_doc:
        raise HTTPException(404, detail="User not found")

    # Security check: dealer_admin can only modify their dealer users
    if current_user.role == "dealer_admin":
        if user_doc.get("dealer_id") != current_user.dealer_id:
            raise HTTPException(403, detail="Not authorized to modify this user.")
    elif current_user.role != "super_admin":
        raise HTTPException(403, detail="Not authorized to modify users.")

    updates = {}
    
    # Check each field and add to updates if provided
    if payload.username is not None:
        # Check if username already exists (excluding current user)
        existing_user = await users_collection.find_one({
            "username": payload.username,
            "_id": {"$ne": ObjectId(user_id)}
        })
        if existing_user:
            raise HTTPException(400, detail="Username already exists")
        updates["username"] = payload.username
        
    if payload.email is not None:
        updates["email"] = payload.email
        
    if payload.password is not None:
        updates["hashed_password"] = get_password_hash(payload.password)
        
    if payload.role is not None and current_user.role == "super_admin":
        # Only super admin may change roles
        updates["role"] = payload.role
        
    if payload.dealer_id is not None and current_user.role == "super_admin":
        # Only super admin may change dealer_id
        updates["dealer_id"] = payload.dealer_id
    if payload.showroom_name is not None:  
        updates["showroom_name"] = payload.showroom_name
    if not updates:
        raise HTTPException(400, detail="No valid fields to update.")

    updates["updated_at"] = dt.utcnow()
    await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": updates})

    # TRANSFER ANALYSIS RESULTS if dealer_id changed (only for super_admin)
    if (payload.dealer_id is not None and 
        current_user.role == "super_admin" and
        user_doc.get("dealer_id") != payload.dealer_id and 
        user_doc.get("dealer_id") is not None):
        
        try:
            # Update all analysis results for this user to the new dealer_id
            result = await results_collection.update_many(
                {
                    "submitted_by_user_id": user_id,
                    "dealer_id": user_doc.get("dealer_id")
                },
                {"$set": {"dealer_id": payload.dealer_id}}
            )
            logger.info(f"Transferred {result.modified_count} analysis results for user {user_id}")
        except Exception as e:
            logger.error(f"Error transferring analysis results: {e}")

    # Return updated user
    user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
    user_doc["_id"] = str(user_doc["_id"])
    return UserInDB(**user_doc)


@app.delete("/users/{user_id}", status_code=204)
async def delete_user(user_id: str, current_user: UserInDB = Depends(get_current_user)):
    """
    Super Admin → can delete any user
    Dealer Admin → can delete only users within their own dealer_id
    """
    if not ObjectId.is_valid(user_id):
        raise HTTPException(400, detail="Invalid user_id")

    user_doc = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user_doc:
        raise HTTPException(404, detail="User not found")

    # Security check
    if current_user.role == "dealer_admin":
        if user_doc.get("dealer_id") != current_user.dealer_id:
            raise HTTPException(403, detail="Not authorized to delete this user.")
    elif current_user.role != "super_admin":
        raise HTTPException(403, detail="Not authorized to delete users.")

    await users_collection.delete_one({"_id": ObjectId(user_id)})
    return Response(status_code=204)



# -----------------------------
# Background Analysis Task Functions
# -----------------------------
async def create_analysis_task(
    citnow_url: str,
    transcription_language: str,
    target_language: str,
    submitted_by_user_id: str,
    dealer_id: Optional[str] = None
) -> str:
    """Create and store analysis task in MongoDB"""
    if analysis_tasks_collection is None:
        raise RuntimeError("Analysis tasks collection not initialized")
        
    task_id = str(uuid.uuid4())
    
    task_data = {
        "task_id": task_id,
        "status": "pending",
        "citnow_url": citnow_url,
        "transcription_language": transcription_language,
        "target_language": target_language,
        "submitted_by_user_id": submitted_by_user_id,
        "dealer_id": dealer_id,
        "created_at": dt.utcnow().isoformat(),
        "updated_at": dt.utcnow().isoformat(),
        "expires_at": (dt.utcnow() + timedelta(hours=24)).isoformat()  # 24h TTL
    }
    
    await analysis_tasks_collection.insert_one(task_data)
    return task_id

@app.get("/users/my-dealer", response_model=List[UserInDB])
async def get_my_dealer_users(current_user: UserInDB = Depends(get_current_user)):
    """
    Get users for the current dealer admin's dealership
    """
    if not current_user.dealer_id:
        raise HTTPException(status_code=400, detail="User is not assigned to a dealer")
    
    if current_user.role not in ["dealer_admin", "super_admin"]:
        raise HTTPException(status_code=403, detail="Not authorized to view dealer users")
    
    users_cursor = users_collection.find({"dealer_id": current_user.dealer_id})
    users_list = await users_cursor.to_list(None)

    for user_doc in users_list:
        user_doc["_id"] = str(user_doc["_id"])
    return [UserInDB(**u) for u in users_list]

async def get_analysis_task(task_id: str) -> Optional[Dict]:
    """Get analysis task from MongoDB"""
    if analysis_tasks_collection is None:
        return None
        
    task = await analysis_tasks_collection.find_one({"task_id": task_id})
    if task:
        task.pop('_id', None)  # Remove MongoDB _id
        return task
    return None

async def update_analysis_task(task_id: str, updates: Dict):
    """Update analysis task in MongoDB"""
    if analysis_tasks_collection is None:
        return
        
    updates["updated_at"] = dt.utcnow().isoformat()
    await analysis_tasks_collection.update_one(
        {"task_id": task_id},
        {"$set": updates}
    )

async def process_single_analysis_task(
    task_id: str,
    citnow_url: str,
    transcription_language: str,
    target_language: str,
    submitted_by_user_id: str,
    dealer_id: Optional[str]
):
    """Background task to process analysis"""
    try:
        # Update task status to processing
        await update_analysis_task(task_id, {
            "status": "processing",
            "message": "Running video analysis..."
        })
        
        # Run the actual analysis
        processed_results = await _run_analysis_pipeline(
            citnow_url,
            transcription_language,
            target_language,
            submitted_by_user_id,
            dealer_id
        )
        
        # Store results
        res = await results_collection.insert_one(processed_results.copy())
        result_id = str(res.inserted_id)
        
        # Update task with success
        await update_analysis_task(task_id, {
            "status": "completed",
            "result_id": result_id,
            "message": "Analysis completed successfully"
        })
        
        logger.info(f"Analysis task {task_id} completed with result {result_id}")
        
    except Exception as e:
        # Update task with error
        error_msg = str(e)
        await update_analysis_task(task_id, {
            "status": "failed",
            "error_message": error_msg,
            "message": f"Analysis failed: {error_msg}"
        })
        
        logger.error(f"Analysis task {task_id} failed: {error_msg}")

# Also add these cleanup functions:
async def cleanup_expired_tasks():
    """Clean up expired tasks (run this periodically)"""
    if analysis_tasks_collection is None:
        return
        
    result = await analysis_tasks_collection.delete_many({
        "expires_at": {"$lt": dt.utcnow().isoformat()}
    })
    if result.deleted_count > 0:
        logger.info(f"Cleaned up {result.deleted_count} expired analysis tasks")

async def periodic_cleanup():
    """Periodically clean up expired tasks"""
    while True:
        try:
            await cleanup_expired_tasks()
            await asyncio.sleep(3600)  # Run every hour
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
            await asyncio.sleep(300) 
# -----------------------------
# Video Analysis Helper Functions (UPDATED for simplified dealer_id)
# -----------------------------

async def _run_analysis_pipeline(
    video_input: str,
    transcription_language: str,
    target_language: str,
    submitted_by_user_id: str,
    dealer_id: Optional[str] # Now a simple string
) -> dict:
    """
    Internal helper to execute the UnifiedMediaAnalyzer pipeline and format results.
    """
    global analyzer
    if analyzer is None:
        logger.error("UnifiedMediaAnalyzer is not initialized.")
        raise RuntimeError("Analysis engine not ready.")

    results, error = await _process_single_video_in_thread(video_input, transcription_language, target_language)

    if error:
        logger.error(f"Analysis pipeline failed for {video_input}: {error}")
        raise RuntimeError(f"Video analysis failed: {error}")

    # Enriched data for storage and dashboarding
    processed_results = {
        "input_source": video_input,
        "processing_timestamp": dt.utcnow().isoformat(),
        "processing_steps": results.get("processing_steps", []),
        "submitted_by_user_id": submitted_by_user_id,
        "dealer_id": dealer_id, # Store as simple string
        "transcription_language": transcription_language,
        "target_language_used": target_language,
        "created_at": dt.utcnow(),
        "status": BatchStatus.COMPLETED,
        
        # Extracted key metrics for dashboard queries
        "overall_quality_score": results.get("overall_quality", {}).get("overall_score"),
        "overall_quality_label": results.get("overall_quality", {}).get("overall_label"),
        "video_quality_score": results.get("video_analysis", {}).get("quality_score"),
        "video_quality_label": results.get("video_analysis", {}).get("quality_label"),
        "audio_quality_score": results.get("audio_analysis", {}).get("score"),
        "audio_clarity_level": results.get("audio_analysis", {}).get("clarity_level"),
        "shake_level": results.get("video_analysis", {}).get("shake_level"),
        "resolution_quality": results.get("video_analysis", {}).get("resolution_quality"),
        "transcription_length": results.get("transcription", {}).get("length"),
        "summary_length": results.get("summarization", {}).get("length"),
        
        # CitNow specific metadata
        "citnow_metadata": results.get("citnow_metadata", {}),
        "citnow_dealership": results.get("citnow_metadata", {}).get("dealership"),
        "citnow_vehicle": results.get("citnow_metadata", {}).get("vehicle"),
        "citnow_registration": results.get("citnow_metadata", {}).get("registration"),
        "citnow_vin": results.get("citnow_metadata", {}).get("vin"),
        "citnow_service_advisor": results.get("citnow_metadata", {}).get("service_advisor"),
        "citnow_brand": results.get("citnow_metadata", {}).get("brand"),
        
        # Store full analysis sub-reports
        "video_analysis": results.get("video_analysis"),
        "audio_analysis": results.get("audio_analysis"),
        "overall_quality": results.get("overall_quality"),
        "transcription": results.get("transcription"),
        "summarization": results.get("summarization"),
        "translation": results.get("translation"),
        "error_message": results.get("error_message"),
    }

    return clean_results(processed_results)

async def _process_single_video_in_thread(video_input: str, transcription_language: str, target_language: str) -> tuple[Optional[dict], Optional[str]]:
    """Executes the CPU-bound video analysis in a separate thread."""
    global analyzer
    if analyzer is None:
        return None, "Analyzer not initialized."

    loop = asyncio.get_running_loop()

    def blocking_analysis():
        with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
            try:
                return analyzer.process_video(
                    video_input,
                    transcription_language=transcription_language,
                    target_language_short=target_language
                )
            except Exception as e:
                logger.error(f"Internal analyzer error for {video_input}: {e}", exc_info=True)
                raise

    try:
        task = loop.run_in_executor(executor, blocking_analysis)
        results = await asyncio.wait_for(task, timeout=PROCESS_TIMEOUT_SECONDS)
        return results, None
    except asyncio.TimeoutError:
        logger.warning(f"Processing timed out for {video_input} after {PROCESS_TIMEOUT_SECONDS} seconds.")
        return None, f"Analysis timed out after {PROCESS_TIMEOUT_SECONDS} seconds."
    except Exception as e:
        logger.warning(f"Processing failed for {video_input}: {e}", exc_info=True)
        return None, str(e)

async def _store_excel_data_in_chunks(batch_id: str, filename: str, df: pd.DataFrame):
    """Stores chunks of Excel data to MongoDB's excel_data_collection."""
    try:
        records = df.to_dict("records")
        chunk_size = 1000
        total_chunks = (len(records) + chunk_size - 1) // chunk_size
        
        insert_operations = []
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            excel_chunk_doc = {
                "batch_id": batch_id,
                "filename": filename,
                "uploaded_at": dt.utcnow(),
                "chunk_index": i // chunk_size,
                "total_chunks": total_chunks,
                "data": chunk,
                "total_rows": len(records)
            }
            insert_operations.append(excel_chunk_doc)
        
        if insert_operations:
            await excel_data_collection.insert_many(insert_operations)
            logger.info("Stored Excel data rows=%d chunks=%d for batch %s.", len(records), total_chunks, batch_id)
    except Exception:
        logger.exception("Could not store Excel data for batch %s.", batch_id)

async def _process_single_batch_url_item(
    batch_id: str, 
    url: str, 
    order: int, 
    transcription_language: str, 
    target_language: str,
    submitted_by_user_id: str,
    dealer_id: Optional[str] = None # Now a simple string
):
    """Processes a single URL within a batch job."""
    
    current_batch_doc = await batch_collection.find_one({"_id": ObjectId(batch_id)})
    if not current_batch_doc or current_batch_doc.get("status") in [BatchStatus.CANCELLED, BatchStatus.STOPPING]:
        logger.info(f"Batch {batch_id} URL {order}: Batch already inactive. Skipping processing of {url[:50]}...")
        return False

    try:
        processed_results = await _run_analysis_pipeline(
            url,
            transcription_language,
            target_language,
            submitted_by_user_id,
            dealer_id
        )
        
        await results_collection.insert_one(processed_results)

        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {
                "$inc": {"processed_urls": 1}, 
                "$set": {"current_url": url, "updated_at": dt.utcnow()}
            }
        )
        logger.info(f"Batch {batch_id}: Successfully processed URL {order} ({url[:50]}...)")
        return True

    except RuntimeError as e:
        logger.warning(f"Batch {batch_id} URL {order} failed during analysis pipeline: {e}")
        error_message = str(e)
        
        error_doc = {
            "batch_id": batch_id,
            "input_source": url,
            "error_message": error_message,
            "processing_order": order,
            "transcription_language": transcription_language,
            "target_language_used": target_language,
            "status": BatchStatus.FAILED,
            "created_at": dt.utcnow(),
            "submitted_by_user_id": submitted_by_user_id,
            "dealer_id": dealer_id
        }
        await results_collection.insert_one(error_doc)
        
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {
                "$inc": {"failed_urls": 1}, 
                "$set": {"current_url": url, "updated_at": dt.utcnow()}
            }
        )
        return False
    except Exception as e:
        logger.exception(f"Batch {batch_id} URL {order}: Unexpected critical error during processing of {url}.")
        error_message = f"Unexpected server error: {str(e)}"
        
        error_doc = {
            "batch_id": batch_id,
            "input_source": url,
            "error_message": error_message,
            "processing_order": order,
            "transcription_language": transcription_language,
            "target_language_used": target_language,
            "status": BatchStatus.FAILED,
            "created_at": dt.utcnow(),
            "submitted_by_user_id": submitted_by_user_id,
            "dealer_id": dealer_id
        }
        await results_collection.insert_one(error_doc)
        
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {
                "$inc": {"failed_urls": 1}, 
                "$set": {"current_url": url, "updated_at": dt.utcnow()}
            }
        )
        return False

async def process_batch_urls_async(
    batch_id: str, 
    urls: List[str], 
    transcription_language: str, 
    target_language: str,
    submitted_by_user_id: str,
    dealer_id: Optional[str] = None # Now a simple string
):
    """Main background task function to manage the processing of all URLs within a batch."""
    
    if analyzer is None:
        logger.error("Analyzer not initialized for batch processing. Batch %s will fail.", batch_id)
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.FAILED, 
                "error": "Analysis engine not ready during batch run.", 
                "updated_at": dt.utcnow()
            }}
        )
        return

    try:
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.PROCESSING, 
                "started_at": dt.utcnow(), 
                "updated_at": dt.utcnow()
            }}
        )
        logger.info(f"Starting batch {batch_id} ({len(urls)} URLs) by user {submitted_by_user_id} for dealer {dealer_id}.")

        for index, url in enumerate(urls):
            if batch_cancellation_flags.get(batch_id, False):
                logger.info(f"Batch {batch_id} cancellation detected. Stopping further URL processing.")
                await batch_collection.update_one(
                    {"_id": ObjectId(batch_id)}, 
                    {"$set": {
                        "status": BatchStatus.CANCELLED, 
                        "updated_at": dt.utcnow()
                    }}
                )
                batch_cancellation_flags.pop(batch_id, None)
                return
            
            await _process_single_batch_url_item(
                batch_id, 
                url, 
                index + 1,
                transcription_language, 
                target_language,
                submitted_by_user_id,
                dealer_id
            )
            
            await asyncio.sleep(0.1)

        batch_cancellation_flags.pop(batch_id, None)
        
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.COMPLETED, 
                "completed_at": dt.utcnow(), 
                "updated_at": dt.utcnow()
            }}
        )
        logger.info(f"Batch {batch_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Batch processing for {batch_id} failed unexpectedly at a high level.")
        batch_cancellation_flags.pop(batch_id, None)
        await batch_collection.update_one(
            {"_id": ObjectId(batch_id)}, 
            {"$set": {
                "status": BatchStatus.FAILED, 
                "error": f"Batch task encountered an unexpected error: {str(e)}", 
                "updated_at": dt.utcnow()
            }}
        )

# -----------------------------
# Analysis Endpoints (UPDATED for simplified dealer_id)
# -----------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video_background(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: UserInDB = Depends(get_current_user)
):
    """Start analysis as background task"""
    try:
        # Create task record in database
        task_id = await create_analysis_task(
            citnow_url=request.citnow_url,
            transcription_language=request.transcription_language,
            target_language=request.target_language,
            submitted_by_user_id=str(current_user.id),
            dealer_id=current_user.dealer_id
        )
        
        # Start background processing
        background_tasks.add_task(
            process_single_analysis_task,
            task_id,
            request.citnow_url,
            request.transcription_language,
            request.target_language,
            str(current_user.id),
            current_user.dealer_id
        )
        
        return {
            "success": True,
            "message": "Analysis started in background",
            "task_id": task_id,  # ADD THIS
            "result_id": None
        }
        
    except Exception as e:
        logger.exception(f"Error starting analysis task: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to start analysis: {str(e)}"
        )
@app.get("/my-analysis-tasks")
async def get_my_analysis_tasks(
    limit: int = 20,
    current_user: UserInDB = Depends(get_current_user)
):
    """Get current user's analysis tasks"""
    if analysis_tasks_collection is None:
        raise HTTPException(status_code=500, detail="Analysis tasks collection not initialized")
        
    tasks_cursor = analysis_tasks_collection.find({
        "submitted_by_user_id": str(current_user.id)
    }).sort("created_at", -1).limit(limit)
    
    tasks = await tasks_cursor.to_list(length=limit)
    
    # Convert ObjectId to string and remove _id
    for task in tasks:
        if '_id' in task:
            task.pop('_id')
    
    return {"tasks": tasks}
@app.get("/analyze-status/{task_id}")
async def get_analysis_status(
    task_id: str, 
    current_user: UserInDB = Depends(get_current_user)
):
    """Get analysis task status with authorization"""
    task = await get_analysis_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found or expired")
    
    # Authorization check
    if (current_user.role in ("dealer_admin","dealer_user") and current_user.dealer_id != task.get("dealer_id")):
        raise HTTPException(
            status_code=403, 
            detail="Not authorized to view this task"
        )
    
    return task

@app.post("/bulk-analyze", response_model=BatchCreateResponse)
async def create_bulk_analysis(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...), 
    transcription_language: str = Form("auto"), 
    target_language: str = Form("en"),
    current_user: UserInDB = Depends(get_current_user)
):
    try:
        if not file.filename or not file.filename.lower().endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are supported.")

        contents = await file.read()
        df = pd.read_excel(_io.BytesIO(contents))
        logger.info("Excel file loaded with %d rows", len(df))

        # Normalize column headers and values for robust detection
        try:
            df.columns = [str(c).strip() for c in df.columns]
        except Exception:
            df.columns = [str(c) for c in df.columns]

        # Try preferred detection by header keywords
        url_column: Optional[str] = None
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['video', 'url', 'link']):
                url_column = col
                break

        urls: list[str] = []
        if url_column is not None and url_column in df.columns:
            series = df[url_column]
            try:
                series = series.astype(str).str.strip()
            except Exception:
                series = series.astype(str)
            urls = series.dropna().unique().tolist()
            urls = [u for u in urls if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]

        # Fallback: pick the column with the most http-links if nothing found
        if not urls:
            best_col = None
            best_count = 0
            for col in df.columns:
                try:
                    col_series = df[col].astype(str)
                except Exception:
                    continue
                count = int(col_series.str.contains('http', case=False, na=False).sum())
                if count > best_count:
                    best_count = count
                    best_col = col
            if best_col is not None and best_count > 0:
                try:
                    urls = df[best_col].astype(str).str.strip().tolist()
                except Exception:
                    urls = df[best_col].astype(str).tolist()
                urls = [u for u in urls if isinstance(u, str) and u.strip().startswith(("http://", "https://"))]
                url_column = best_col

        if not urls:
            raise HTTPException(status_code=400, detail="No valid URLs found in the Excel file. Ensure a column contains full http(s) links.")

        logger.info("Found %d unique URLs to process", len(urls))

        batch_job = {
            "status": BatchStatus.PENDING,
            "total_urls": len(urls),
            "processed_urls": 0,
            "failed_urls": 0,
            "urls": urls,
            "transcription_language": transcription_language,
            "target_language": target_language,
            "original_filename": file.filename,
            "created_at": dt.utcnow(),
            "updated_at": dt.utcnow(),
            "submitted_by_user_id": str(current_user.id),
            "dealer_id": current_user.dealer_id, # Now a simple string
        }
        
        inserted = await batch_collection.insert_one(batch_job)
        batch_id = str(inserted.inserted_id)
        
        logger.info(f"Created new batch: {batch_id} with {len(urls)} URLs by user {current_user.username}.")

        batch_cancellation_flags[batch_id] = False
        await _store_excel_data_in_chunks(batch_id, file.filename, df)

        background_tasks.add_task(
            process_batch_urls_async, 
            batch_id, 
            urls, 
            transcription_language, 
            target_language,
            str(current_user.id),
            current_user.dealer_id # Now a simple string
        )

        return {
            "success": True, 
            "batch_id": batch_id, 
            "total_urls": len(urls), 
            "message": f"Batch processing started for {len(urls)} URLs."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /bulk-analyze endpoint")
        raise HTTPException(status_code=500, detail=f"Error processing Excel file: {str(e)}")

# -----------------------------
# Batch Control Endpoints (UPDATED for simplified dealer_id)
# -----------------------------
@app.post("/bulk-cancel/{batch_id}")
async def cancel_bulk_processing(batch_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    object_id = ObjectId(batch_id)
    
    batch = await batch_collection.find_one({"_id": object_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    
    # Authorization: Super Admin can cancel any batch; Dealer Admin can only cancel their own dealer's batch.
    if current_user.role == "dealer_admin" and current_user.dealer_id != batch.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to cancel this batch.")
    
    current_status = batch.get("status")
    if current_status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]:
        return {
            "success": False, 
            "message": f"Cannot cancel batch with status: '{current_status}'.",
            "current_status": current_status
        }
    
    batch_cancellation_flags[batch_id] = True
    
    update_result = await batch_collection.update_one(
        {"_id": object_id}, 
        {"$set": {
            "status": BatchStatus.STOPPING,
            "cancelled_at": dt.utcnow(),
            "updated_at": dt.utcnow()
        }}
    )
    
    if update_result.modified_count == 0:
        logger.warning(f"Batch {batch_id} status update to STOPPING failed.")
    
    logger.info(f"Batch {batch_id} cancellation requested by {current_user.username}. Previous status: {current_status}")
    return {
        "success": True, 
        "message": "Batch cancellation initiated. Processing will stop shortly.",
        "batch_id": batch_id,
        "previous_status": current_status
    }

@app.delete("/bulk-job/{batch_id}")
async def delete_bulk_job(batch_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    object_id = ObjectId(batch_id)
    
    batch = await batch_collection.find_one({"_id": object_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    
    # Authorization: Super Admin can delete any batch; Dealer Admin can only delete their own dealer's batch.
    if current_user.role == "dealer_admin" and current_user.dealer_id != batch.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this batch.")
    
    batch_cancellation_flags[batch_id] = True
    
    delete_results = await results_collection.delete_many({"batch_id": batch_id})
    delete_excel_data = await excel_data_collection.delete_many({"batch_id": batch_id})
    delete_batch = await batch_collection.delete_one({"_id": object_id})
    
    if delete_batch.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Batch not found during deletion.")
    
    batch_cancellation_flags.pop(batch_id, None)
    
    logger.info(f"Deleted batch {batch_id} by {current_user.username}: {delete_results.deleted_count} analysis results, {delete_excel_data.deleted_count} excel chunks.")
    
    return {
        "success": True,
        "message": f"Batch '{batch_id}' and all {delete_results.deleted_count} associated results deleted successfully.",
        "deleted_results": delete_results.deleted_count,
        "deleted_excel_chunks": delete_excel_data.deleted_count,
        "batch_id": batch_id
    }
    
@app.get("/bulk-batches", response_model=List[BatchCreateResponse])
async def list_all_batches(limit: int = 50, status_filter: Optional[str] = None, current_user: UserInDB = Depends(get_current_user)):
    query: dict = {}
    if status_filter:
        if status_filter not in [s.value for s in BatchStatus]:
            raise HTTPException(status_code=400, detail=f"Invalid status_filter. Must be one of: {[s.value for s in BatchStatus]}.")
        query["status"] = status_filter
    
    # Authorization: Filter batches by dealer_id for Dealer Admins. Super Admins see all.
    if current_user.role == "dealer_admin" and current_user.dealer_id:
        query["dealer_id"] = current_user.dealer_id # Simple string comparison
    
    batches_cursor = batch_collection.find(query).sort("created_at", -1)
    batches = await batches_cursor.to_list(min(limit, 100))
    
    result = []
    for batch in batches:
        result.append(BatchCreateResponse(
            success=True,                   # <-- ADD THIS
            batch_id=str(batch["_id"]),
            status=batch.get("status"),
            total_urls=batch.get("total_urls", 0),
            processed_urls=batch.get("processed_urls", 0),
            failed_urls=batch.get("failed_urls", 0),
            created_at=batch.get("created_at"),
            updated_at=batch.get("updated_at"),
            filename=batch.get("original_filename", "Unknown"),
            submitted_by_user_id=batch.get("submitted_by_user_id"),
            dealer_id=batch.get("dealer_id"),
            message=f"Batch {str(batch['_id'])} status: {batch.get('status')}"
))

    
    return result
    
@app.post("/bulk-stop-all")
async def stop_all_processing(current_user: UserInDB = Depends(get_current_super_admin)):
    processing_batches_cursor = batch_collection.find({
        "status": {"$in": [BatchStatus.PROCESSING, BatchStatus.PENDING]}
    })
    
    stopped_count = 0
    async for batch in processing_batches_cursor:
        batch_id = str(batch["_id"])
        batch_cancellation_flags[batch_id] = True
        
        await batch_collection.update_one(
            {"_id": batch["_id"]}, 
            {"$set": {
                "status": BatchStatus.STOPPING,
                "cancelled_at": dt.utcnow(),
                "updated_at": dt.utcnow()
            }}
        )
        stopped_count += 1
    
    logger.info(f"Super Admin {current_user.username} requested to stop all processing. Signaled {stopped_count} batches.")
    return {"success": True, "message": f"Stopping {stopped_count} active batch(es)."}

# -----------------------------
# Status & Results Endpoints (UPDATED for simplified dealer_id)
# -----------------------------
@app.get("/bulk-status/{batch_id}", response_model=BatchStatusResponse)
async def get_bulk_status(batch_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    object_id = ObjectId(batch_id)
    
    batch = await batch_collection.find_one({"_id": object_id})
    if not batch:
        raise HTTPException(status_code=404, detail=f"Batch not found: {batch_id}")
    
    # Authorization: Super Admin can view any batch; Dealer Admin can only view their own dealer's batch.
    if current_user.role == "dealer_admin" and current_user.dealer_id != batch.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this batch's status.")

    processed = batch.get("processed_urls", 0)
    total = batch.get("total_urls", 0)
    progress = (processed / total * 100) if total > 0 else 0
    current_status = batch.get("status", "unknown")
    
    can_cancel = current_status in [BatchStatus.PENDING, BatchStatus.PROCESSING]
    
    return {
        "batch_id": batch_id,
        "status": current_status,
        "total_urls": total,
        "processed_urls": processed,
        "failed_urls": batch.get("failed_urls", 0),
        "progress_percentage": round(progress, 2),
        "current_url": batch.get("current_url"),
        "can_cancel": can_cancel
    }

@app.get("/bulk-results/{batch_id}")
async def get_bulk_results(batch_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    object_id = ObjectId(batch_id)

    batch = await batch_collection.find_one({"_id": object_id})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")

    if current_user.role == "dealer_admin" and current_user.dealer_id != batch.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this batch's results.")

    results = await results_collection.find({"batch_id": batch_id}).sort("created_at", -1).to_list(None)
    
    return {
        "batch_id": batch_id, 
        "status": batch.get("status"), 
        "total_processed": len(results), 
        "results": [clean_results(r) for r in results]
    }

@app.get("/bulk-download/{batch_id}/structured")
async def download_structured_results(batch_id: str, response: Response, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    object_id = ObjectId(batch_id)

    batch_doc = await batch_collection.find_one({"_id": object_id})
    if not batch_doc:
        raise HTTPException(status_code=404, detail="Batch not found.")
    
    if current_user.role == "dealer_admin" and current_user.dealer_id != batch_doc.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to download results for this batch.")

    temp_dir = tempfile.mkdtemp()
    batch_output_root = os.path.join(temp_dir, batch_id)
    os.makedirs(batch_output_root, exist_ok=True)
    
    results_cursor = results_collection.find({"batch_id": batch_id})
    
    num_results = 0
    async for result in results_cursor:
        num_results += 1
        dealership = result.get("citnow_dealership") or result.get("citnow_metadata", {}).get("dealership", "Unknown_Dealership")
        sanitized_dealer_name = _sanitize_path_segment(dealership)
        
        dealer_dir = os.path.join(batch_output_root, sanitized_dealer_name)
        os.makedirs(dealer_dir, exist_ok=True)
        
        original_url = result.get("input_source", "unknown_url")
        url_hash = hashlib.md5(original_url.encode()).hexdigest()[:8]
        url_segment_match = re.search(r'[^/]+(?=\.mp4$|$)', original_url)
        base_filename = url_segment_match.group(0) if url_segment_match else url_hash
        safe_base_filename = _sanitize_path_segment(base_filename)
        
        report_name_prefix = f"analysis_{safe_base_filename}"
        
        # Save JSON report
        json_filename = f"{report_name_prefix}_{str(result['_id'])}.json"
        json_path = os.path.join(dealer_dir, json_filename)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_results(result), f, ensure_ascii=False, indent=2)

        # Generate and save TXT report
        if analyzer is None:
            raise RuntimeError("UnifiedMediaAnalyzer not initialized for report generation.")
        
        txt_report_content = analyzer.generate_comprehensive_report(clean_results(result))
        txt_filename = f"{report_name_prefix}_{str(result['_id'])}.txt"
        txt_path = os.path.join(dealer_dir, txt_filename)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_report_content)

    if num_results == 0:
        shutil.rmtree(temp_dir)
        raise HTTPException(status_code=404, detail="No analysis results found for this batch.")

    zip_filename = f"batch_{batch_id}_structured_reports.zip"
    zip_filepath = os.path.join(temp_dir, zip_filename)
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(batch_output_root):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, temp_dir))

    response.headers["Content-Disposition"] = f"attachment; filename=\"{zip_filename}\""

    return FileResponse(
        path=zip_filepath,
        filename=zip_filename,
        media_type="application/zip",
        background=BackgroundTasks(lambda: shutil.rmtree(temp_dir))
    )

@app.get("/bulk-excel-data/{batch_id}")
async def get_bulk_excel_data(batch_id: str, chunk: int = 0, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(batch_id):
        raise HTTPException(status_code=400, detail="Invalid batch ID format.")
    
    batch = await batch_collection.find_one({"_id": ObjectId(batch_id)})
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found.")
    
    if current_user.role == "dealer_admin" and current_user.dealer_id != batch.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this batch's excel data.")

    excel_data = await excel_data_collection.find_one({"batch_id": batch_id, "chunk_index": chunk})
    if not excel_data:
        excel_data_check = await excel_data_collection.find_one({"batch_id": batch_id})
        if not excel_data_check:
            raise HTTPException(status_code=404, detail="Excel data not found for this batch.")
        else:
            raise HTTPException(status_code=404, detail=f"Excel data chunk {chunk} not found for this batch. Max chunk index is {excel_data_check.get('total_chunks', 1) - 1}.")

    return {
        "batch_id": str(excel_data.get("batch_id")),
        "filename": excel_data.get("filename"), 
        "uploaded_at": excel_data.get("uploaded_at"), 
        "total_rows": excel_data.get("total_rows"), 
        "chunk_index": excel_data.get("chunk_index", 0), 
        "total_chunks": excel_data.get("total_chunks", 1), 
        "data": excel_data.get("data", [])
    }

@app.get("/results", response_class=ORJSONResponse)
async def get_all_results(
batch_id: Optional[str] = None,
dealer_id: Optional[str] = None,
limit: int = 100,
current_user: UserInDB = Depends(get_current_user)
):
    limit = max(1, min(limit, 1000))
    query: Dict[str, Any] = {}

    if batch_id:
        if not ObjectId.is_valid(batch_id):
            raise HTTPException(status_code=400, detail="Invalid batch ID format.")
        query["batch_id"] = batch_id

    # RBAC scoping
    if current_user.role == "super_admin":
        if dealer_id:
            query["dealer_id"] = dealer_id
    elif current_user.role in ("dealer_admin", "dealer_user"):
        if not current_user.dealer_id:
            raise HTTPException(status_code=403, detail="User has no assigned dealer_id.")
        query["dealer_id"] = current_user.dealer_id
    else:
        raise HTTPException(status_code=403, detail="Not authorized to view results")

    # Fetch and return
    results = await results_collection.find(query).sort("created_at", -1).limit(limit).to_list(length=limit)
    return [clean_results(r) for r in results]
    
    ''''elif current_user.role == "dealer_user":
        query["submitted_by_user_id"] = str(current_user.id)'''



@app.get("/results/{result_id}")
async def get_result(result_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(result_id):
        raise HTTPException(status_code=400, detail="Invalid Result ID format.")

    result = await results_collection.find_one({"_id": ObjectId(result_id)})
    if not result:
        raise HTTPException(status_code=404, detail="Analysis result not found.")

    if current_user.role in ("dealer_admin", "dealer_user") and current_user.dealer_id != result.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to view this result.")

    return clean_results(result)

@app.delete("/results/{result_id}")
async def delete_result(result_id: str, current_user: UserInDB = Depends(get_current_user)):
    if not ObjectId.is_valid(result_id):
        raise HTTPException(status_code=400, detail="Invalid Result ID format.")

    result = await results_collection.find_one({"_id": ObjectId(result_id)})
    if not result:
        raise HTTPException(status_code=404, detail="Result not found.")

    if current_user.role in ("dealer_admin", "dealer_user") and current_user.dealer_id != result.get("dealer_id"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to delete this result.")

    delete_operation = await results_collection.delete_one({"_id": ObjectId(result_id)})
    if delete_operation.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Result not found for deletion (may have been already deleted).")

    return JSONResponse(status_code=200, content={"success": True, "message": "Result deleted successfully."})

# -----------------------------
# Dashboard Endpoints (UPDATED for simplified dealer_id)
# -----------------------------

@app.get("/dashboard/super-admin/overview", response_model=SuperAdminDashboardOverview)
async def get_super_admin_dashboard_overview(current_user: UserInDB = Depends(get_current_super_admin)):
    """
    Retrieves aggregated data for the Super Admin dashboard.
    """
    total_videos = await results_collection.count_documents({"status": BatchStatus.COMPLETED})
    
    avg_overall_quality_agg = await results_collection.aggregate([
        {"$match": {"overall_quality_score": {"$exists": True, "$ne": None}, "status": BatchStatus.COMPLETED}},
        {"$group": {"_id": None, "average": {"$avg": "$overall_quality_score"}}}
    ]).to_list(1)

    avg_overall_quality = avg_overall_quality_agg[0]["average"] if avg_overall_quality_agg else 0

    quality_distribution_raw = await results_collection.aggregate([
        {"$match": {"status": BatchStatus.COMPLETED}},
        {"$group": {"_id": "$overall_quality_label", "count": {"$sum": 1}}}
    ]).to_list(None)
    quality_distribution = {item['_id']: item['count'] for item in quality_distribution_raw if item['_id']}

    # SIMPLIFIED: No dealer name lookup, just use dealer_id directly
    dealers_summary_raw = await results_collection.aggregate([
        {"$match": {"dealer_id": {"$exists": True, "$ne": None}, "status": BatchStatus.COMPLETED}},
        {"$group": {
            "_id": "$dealer_id",
            "total_videos": {"$sum": 1},
            "avg_overall_quality": {"$avg": "$overall_quality_score"}
        }},
        {"$project": {
            "dealer_id": "$_id", # Use dealer_id directly as string
            "total_videos": 1,
            "avg_overall_quality": {"$round": ["$avg_overall_quality", 1]}
        }}
    ]).to_list(None)
    dealers_summary = [DealerSummary(**d) for d in dealers_summary_raw]

    return SuperAdminDashboardOverview(
        total_videos_analyzed=total_videos,
        average_overall_quality=round(avg_overall_quality, 1),
        quality_distribution=quality_distribution,
        dealers_summary=dealers_summary,
        last_updated=dt.utcnow()
    )


@app.get("/dashboard/dealer/overview", response_model=DealerAdminDashboardOverview)
async def get_dealer_dashboard_overview(current_user: UserInDB = Depends(get_current_dealer_admin)):
    """
    Retrieves aggregated data for a specific Dealer Admin's dashboard.
    """
    if not current_user.dealer_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Dealer Admin is not assigned to a dealer.")

    dealer_id_str = current_user.dealer_id
    
    total_videos = await results_collection.count_documents({"dealer_id": dealer_id_str, "status": BatchStatus.COMPLETED})

    avg_overall_quality_agg = await results_collection.aggregate([
        {"$match": {"dealer_id": dealer_id_str, "overall_quality_score": {"$exists": True, "$ne": None}, "status": BatchStatus.COMPLETED}},
        {"$group": {"_id": None, "average": {"$avg": "$overall_quality_score"}}}
    ]).to_list(1)
    avg_overall_quality = avg_overall_quality_agg[0]["average"] if avg_overall_quality_agg else 0

    quality_distribution_raw = await results_collection.aggregate([
        {"$match": {"dealer_id": dealer_id_str, "status": BatchStatus.COMPLETED}},
        {"$group": {"_id": "$overall_quality_label", "count": {"$sum": 1}}}
    ]).to_list(None)
    quality_distribution = {item['_id']: item['count'] for item in quality_distribution_raw if item['_id']}

    low_quality_videos = await results_collection.count_documents({
        "dealer_id": dealer_id_str,
        "status": BatchStatus.COMPLETED,
        "video_quality_label": {"$in": ["Poor", "Very Poor", "Analysis Failed", "Error"]}
    })
    low_quality_audio = await results_collection.count_documents({
        "dealer_id": dealer_id_str,
        "status": BatchStatus.COMPLETED,
        "audio_clarity_level": {"$in": ["Poor", "Very Poor", "Unusable", "Analysis Failed", "No Audio"]}
    })

    recent_videos_raw = await results_collection.find(
        {"dealer_id": dealer_id_str, "status": BatchStatus.COMPLETED}
    ).sort("created_at", -1).limit(5).to_list(5)
    recent_analyses = [RecentAnalysis(**clean_results(r)) for r in recent_videos_raw]

    return DealerAdminDashboardOverview(
        dealer_id=dealer_id_str,
        total_videos_analyzed=total_videos,
        average_overall_quality=round(avg_overall_quality, 1),
        quality_distribution=quality_distribution,
        low_quality_video_count=low_quality_videos,
        low_quality_audio_count=low_quality_audio,
        recent_analyses=recent_analyses,
        last_updated=dt.utcnow()
    )



@app.get("/dashboard/dealer/{dealer_id}/user-stats")
async def get_dealer_user_stats(
    dealer_id: str, 
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get video analysis statistics for all users in a dealer
    """
    # Authorization check
    if current_user.role == "dealer_admin" and current_user.dealer_id != dealer_id:
        raise HTTPException(status_code=403, detail="Not authorized to view this dealer's user stats")
    
    # Get all users for this dealer
    users_cursor = users_collection.find({"dealer_id": dealer_id})
    users = await users_cursor.to_list(None)
    
    user_stats = []
    for user in users:
        user_id = str(user["_id"])
        
        # Count videos analyzed by this user
        video_count = await results_collection.count_documents({
            "submitted_by_user_id": user_id,
            "dealer_id": dealer_id,
            "status": BatchStatus.COMPLETED
        })
        
        user_stats.append({
            "user_id": user_id,
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "videos_analyzed": video_count
        })
    
    return user_stats

# Add this new endpoint to your main.py (around the existing /users endpoint)

@app.get("/users/by-dealer/{dealer_id}", response_model=List[UserInDB])
async def get_users_by_dealer(
    dealer_id: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """
    Get users for a specific dealer_id
    Super Admin → can view users for any dealer
    Dealer Admin → can only view users for their own dealer
    """
    # Authorization check
    if current_user.role == "dealer_admin" and current_user.dealer_id != dealer_id:
        raise HTTPException(status_code=403, detail="Not authorized to view users for this dealer")
    
    users_cursor = users_collection.find({"dealer_id": dealer_id})
    users_list = await users_cursor.to_list(None)

    for user_doc in users_list:
        user_doc["_id"] = str(user_doc["_id"])
    return [UserInDB(**u) for u in users_list]

# -----------------------------
# Health Check & Root Endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        await client.admin.command('ping')
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": dt.utcnow().isoformat(),
        "database": db_status,
        "active_concurrent_analysis_slots": MAX_WORKERS - executor._work_queue.qsize(),
        "running_batch_tasks_signaled": len(batch_cancellation_flags),
        "analyzer_ready": analyzer is not None
    }

@app.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {
        "message": f"{APP_TITLE} v{APP_VERSION}",
        "description": "API for video analysis, transcription, summarization, and translation with RBAC and dashboard capabilities.",
        "endpoints": {
            "Auth & Users": {
                "Login": "/token (POST)",
                "My Profile": "/users/me (GET)",
                "Create User (Super Admin)": "/users/ (POST)",
                "List Users (Super Admin)": "/users/ (GET)",
            },
            "Video Analysis": {
                "Single Analysis (Background)": "/analyze (POST)",
                "Check Analysis Status": "/analyze-status/{task_id} (GET)",
                "My Analysis Tasks": "/my-analysis-tasks (GET)",
                "Bulk Analysis (Upload Excel)": "/bulk-analyze (POST)", 
                "Get Batch Status": "/bulk-status/{batch_id} (GET)",
                "Cancel Batch": "/bulk-cancel/{batch_id} (POST)",
                "Delete Batch": "/bulk-job/{batch_id} (DELETE)",
                "Download Structured Reports": "/bulk-download/{batch_id}/structured (GET)",
                "Get Excel Data": "/bulk-excel-data/{batch_id} (GET)",
                "List All Results": "/results (GET)",
                "Get Single Result": "/results/{result_id} (GET)",
                "Delete Single Result": "/results/{result_id} (DELETE)",
            },
            "Dashboards": {
                "Super Admin Overview": "/dashboard/super-admin/overview (GET)",
                "Dealer Admin Overview": "/dashboard/dealer/overview (GET)",
            },
            "System": {
                "Health Check": "/health (GET)",
                "API Root": "/ (GET)",
            }
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
