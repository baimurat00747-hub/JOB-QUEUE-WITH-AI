import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

BEARER = HTTPBearer(auto_error=False)
AUDIENCE = "queue-agent"
ROLES = {"viewer", "operator"}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str


def auth_secret():
    return os.environ.get("QUEUE_AGENT_JWT_SECRET", "")


def require_operator(principal: Principal):
    if principal.role != "operator":
        raise HTTPException(403, "Operator access is required.")


def authenticate(credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER)]):
    secret = auth_secret()
    if len(secret) < 32:
        raise HTTPException(503, "Set QUEUE_AGENT_JWT_SECRET to enable authenticated workflows.")
    if credentials is not None:
        try:
            claims = jwt.decode(
                credentials.credentials, secret, algorithms=["HS256"], audience=AUDIENCE, issuer=AUDIENCE,
                options={"require": ["sub", "role", "exp", "iat", "iss", "aud"]},
            )
            if claims["role"] in ROLES and claims["sub"]:
                return Principal(claims["sub"], claims["role"])
        except (jwt.InvalidTokenError, TypeError):
            pass
    raise HTTPException(401, "A valid bearer token is required.", headers={"WWW-Authenticate": "Bearer"})


def legacy_read(credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER)]):
    if auth_secret():
        return authenticate(credentials)


def legacy_write(credentials: Annotated[HTTPAuthorizationCredentials | None, Security(BEARER)]):
    if auth_secret():
        principal = authenticate(credentials)
        require_operator(principal)
        return principal


def signed_token(subject, role):
    now = datetime.now(timezone.utc)
    return jwt.encode({
        "sub": subject, "role": role, "iat": now, "exp": now + timedelta(hours=1),
        "iss": AUDIENCE, "aud": AUDIENCE,
    }, auth_secret(), algorithm="HS256")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Issue a short-lived local demo token.")
    parser.add_argument("role", choices=sorted(ROLES))
    parser.add_argument("subject")
    args = parser.parse_args()
    if len(auth_secret()) < 32:
        parser.error("set QUEUE_AGENT_JWT_SECRET to a random value of at least 32 characters")
    print(signed_token(args.subject, args.role))
