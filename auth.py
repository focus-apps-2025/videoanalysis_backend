from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from datetime import datetime, timedelta
from database import get_collection
from utils import verify_password, get_password_hash
from config import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from typing import Dict, Any

users_coll = get_collection("users")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
router = APIRouter(tags=["authentication"])  # Add tags for better API docs

def authenticate_user(email: str, password: str) -> Dict[str, Any]:
    """Authenticate user by email and password"""
    user = users_coll.find_one({"email": email})
    if not user or not verify_password(password, user["password_hash"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: timedelta = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = users_coll.find_one({"email": email})
    if user is None:
        raise credentials_exception
    return user

def require_roles(allowed_roles: list):
    """Role-Based Access Control dependency"""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)):
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient privileges"
            )
        return user
    return role_checker

@router.post("/login", response_model=Dict[str, str])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint"""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    token_data = {
        "sub": user["email"],
        "role": user["role"],
        "dealer_id": str(user.get("dealer_id", "")),
        "user_id": str(user["_id"])
    }
    access_token = create_access_token(token_data)
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    email: str, 
    password: str, 
    role: str = "dealer", 
    dealer_id: str = None, 
    current_user: Dict[str, Any] = Depends(require_roles(["superadmin"]))
):
    """Register new user (superadmin only)"""
    if users_coll.find_one({"email": email}):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    user_data = {
        "email": email,
        "password_hash": get_password_hash(password),
        "role": role,
        "dealer_id": dealer_id,
        "created_at": datetime.utcnow()
    }
    
    result = users_coll.insert_one(user_data)
    return {
        "success": True, 
        "message": "User registered successfully",
        "user_id": str(result.inserted_id)
    }