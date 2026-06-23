"""Shared FastAPI dependencies for auth + RBAC.

``get_principal`` is the single source of tenant identity:
- ``AUTH_ENABLED`` true  -> identity comes from the verified JWT (header ignored).
- ``AUTH_ENABLED`` false -> falls back to the trusted ``X-Tenant-ID`` header with
  a full-access role, preserving the original demo/test behaviour.

``require(action)`` builds a dependency that additionally enforces RBAC when auth
is enabled.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer

from backend.config import settings
from backend.services.auth import Principal, decode_token, role_allows

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token", auto_error=False)


def get_principal(
    token: Optional[str] = Depends(oauth2_scheme),
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
) -> Principal:
    if settings.AUTH_ENABLED:
        if not token:
            raise HTTPException(status_code=401, detail="Authentication required")
        return decode_token(token)
    # Legacy fallback: trust the header, grant full role.
    return Principal(tenant_id=x_tenant_id or "", role="admin")


def require(action: str):
    """Dependency factory enforcing that the principal's role permits ``action``
    (only when auth is enabled; a no-op authz in fallback mode)."""

    def _dep(principal: Principal = Depends(get_principal)) -> Principal:
        if settings.AUTH_ENABLED and not role_allows(principal.role, action):
            raise HTTPException(
                status_code=403,
                detail=f"Role '{principal.role}' is not permitted to {action}",
            )
        return principal

    return _dep
