"""Bearer-token authentication dependency for the amorphouspy API."""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import API_TOKEN

# When API_TOKEN is not set, the scheme is optional (no lock icon, no enforcement).
# When set, the scheme is required and Swagger shows the Authorize button.
_scheme = HTTPBearer(auto_error=bool(API_TOKEN))


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_scheme)],
) -> None:
    """Validate the bearer token if API_TOKEN is configured."""
    if not API_TOKEN:
        return

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    if not secrets.compare_digest(credentials.credentials, API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid bearer token",
        )
