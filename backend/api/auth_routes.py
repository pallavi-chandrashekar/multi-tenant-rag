"""Authentication endpoints: bootstrap a user and exchange credentials for a JWT.

``/auth/register`` is an open bootstrap endpoint for the reference deployment (in
production it would be admin-gated or replaced by an IdP). ``/auth/token`` is the
OAuth2 password flow used by Swagger UI and the frontend.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models import User
from backend.services.auth import (
    ROLE_PERMISSIONS,
    create_access_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/config")
def auth_config():
    """Public: lets the frontend decide whether to show the login gate."""
    return {"auth_enabled": settings.AUTH_ENABLED}


class RegisterPayload(BaseModel):
    username: str
    password: str
    tenant_id: str
    role: str = "viewer"


@router.post("/register")
def register(payload: RegisterPayload, db: Session = Depends(get_db)):
    if payload.role not in ROLE_PERMISSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"role must be one of {sorted(ROLE_PERMISSIONS)}",
        )
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already exists")

    user = User(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        tenant_id=payload.tenant_id,
        role=payload.role,
    )
    db.add(user)
    db.commit()
    return {"status": "success", "username": user.username, "tenant_id": user.tenant_id, "role": user.role}


@router.post("/token")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(
        tenant_id=user.tenant_id, role=user.role, username=user.username
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "tenant_id": user.tenant_id,
        "role": user.role,
    }
