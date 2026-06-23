"""Authentication (JWT) and role-based authorization.

When ``AUTH_ENABLED`` is true, tenant identity is taken from a verified JWT
claim (issued by ``/auth/token``) instead of the trusted ``X-Tenant-ID`` header,
and ``require_role`` enforces action-level authorization. When disabled, the
endpoints fall back to the header so the demo and tests keep working.

``jose`` / ``passlib`` are imported lazily so the pure RBAC logic
(``role_allows``) stays importable for unit tests without those dependencies.
"""

import datetime
from dataclasses import dataclass
from typing import Optional

# Role -> the set of actions it may perform. Higher roles are supersets.
ROLE_PERMISSIONS = {
    "viewer": {"query", "eval"},
    "editor": {"query", "eval", "ingest"},
    "admin": {"query", "eval", "ingest", "delete", "manage"},
}


@dataclass
class Principal:
    """The authenticated caller: which tenant they act for and their role."""

    tenant_id: str
    role: str = "admin"
    username: Optional[str] = None


def role_allows(role: str, action: str) -> bool:
    """Pure RBAC check: may ``role`` perform ``action``? (unit-testable)."""
    return action in ROLE_PERMISSIONS.get(role, set())


# -- Password hashing (lazy passlib) ---------------------------------------
def _pwd_context():
    from passlib.context import CryptContext

    return CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return _pwd_context().hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return _pwd_context().verify(password, hashed)


# -- Token issue / verify (lazy jose) --------------------------------------
def create_access_token(tenant_id: str, role: str, username: str) -> str:
    """Mint a signed JWT whose claims carry the tenant + role."""
    from jose import jwt

    from backend.config import settings

    expire = datetime.datetime.utcnow() + datetime.timedelta(
        minutes=settings.JWT_EXPIRE_MINUTES
    )
    payload = {
        "sub": username,
        "tenant_id": tenant_id,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> Principal:
    """Verify a JWT and return the Principal, or raise 401 on any failure."""
    from fastapi import HTTPException
    from jose import JWTError, jwt

    from backend.config import settings

    try:
        claims = jwt.decode(
            token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM]
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    tenant_id = claims.get("tenant_id")
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Token missing tenant claim")
    return Principal(
        tenant_id=tenant_id,
        role=claims.get("role", "viewer"),
        username=claims.get("sub"),
    )
