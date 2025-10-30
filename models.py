from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)  # Add password validation
    role: str = "dealer"
    dealer_id: Optional[str] = None

class UserOut(BaseModel):
    email: EmailStr
    role: str
    dealer_id: Optional[str] = None
    created_at: Optional[datetime] = None  # Add if you return this field

class Token(BaseModel):
    access_token: str
    token_type: str

class LoginData(BaseModel):
    username: EmailStr
    password: str

# Optional: Add token data model
class TokenData(BaseModel):
    email: Optional[str] = None
    role: Optional[str] = None
    dealer_id: Optional[str] = None