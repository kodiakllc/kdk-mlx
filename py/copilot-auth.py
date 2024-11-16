import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# In-memory store for tokens and user info
tokens = {
    "mock_access_token": {
        "user": {
            "login": "mock_user",
            "id": 123456,
            "node_id": "MDQ6VXNlcjE=",
            "avatar_url": "https://example.com/avatar.png",
            "gravatar_id": "",
            "url": "https://api.example.com/users/mock_user",
            "html_url": "https://example.com/mock_user",
            "followers_url": "https://api.example.com/users/mock_user/followers",
            "following_url": "https://api.example.com/users/mock_user/following{/other_user}",
            "gists_url": "https://api.example.com/users/mock_user/gists{/gist_id}",
            "starred_url": "https://api.example.com/users/mock_user/starred{/owner}{/repo}",
            "subscriptions_url": "https://api.example.com/users/mock_user/subscriptions",
            "organizations_url": "https://api.example.com/users/mock_user/orgs",
            "repos_url": "https://api.example.com/users/mock_user/repos",
            "events_url": "https://api.example.com/users/mock_user/events{/privacy}",
            "received_events_url": "https://api.example.com/users/mock_user/received_events",
            "type": "User",
            "site_admin": False,
            "name": "Mock User",
            "company": "Example",
            "blog": "https://example.com/blog",
            "location": "Earth",
            "email": "mock_user@example.com",
            "hireable": True,
            "bio": "This is a mock user.",
            "twitter_username": "mockuser",
            "public_repos": 2,
            "public_gists": 1,
            "followers": 0,
            "following": 0,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
    }
}

class TokenRequest(BaseModel):
    client_id: str
    device_code: str
    grant_type: str

@app.middleware("http")
async def log_request_info(request: Request, call_next):
    logger.debug(f'Headers: {request.headers}')
    logger.debug(f'Body: {await request.body()}')
    response = await call_next(request)
    return response

# Mock Device Authorization Endpoint
@app.post('/login/device/code')
async def device_authorization():
    response = {
        "device_code": "mock_device_code",
        "user_code": "mock_user_code",
        "verification_uri": "http://localhost:5201/device",
        "verification_uri_complete": "http://localhost:5201/device?user_code=mock_user_code",
        "expires_in": 900,
        "interval": 5
    }
    return JSONResponse(content=response)

# Mock Verification URI Endpoint
@app.get('/device')
async def device_verification(user_code: Optional[str] = "mock_user_code"):
    return JSONResponse(content={"message": f"Enter the user code: {user_code}. Verification successful."})

# Mock Token Endpoint for POST requests
@app.post('/login/oauth/access_token')
async def token(request: TokenRequest):
    logger.debug(f'Client ID: {request.client_id}')
    logger.debug(f'Device Code: {request.device_code}')
    logger.debug(f'Grant Type: {request.grant_type}')

    if not request.client_id or not request.device_code or not request.grant_type:
        raise HTTPException(status_code=400, detail="invalid_request")

    if request.device_code == "mock_device_code":
        response = {
            "access_token": "mock_access_token",
            "token_type": "bearer",
            "expires_in": 3600,
            "refresh_token": "mock_refresh_token"
        }
        return JSONResponse(content=response)
    else:
        raise HTTPException(status_code=400, detail="invalid_grant")

# Mock GitHub User Endpoint
@app.get('/user')
async def get_user():
    response = {
        "id": 123456,
        "login": "mock_user"
    }
    return JSONResponse(content=response)

# Mock User Info Endpoint for GitHub Enterprise
@app.get('/api/v3/user')
async def github_enterprise_user_info(request: Request):
    auth_header = request.headers.get('Authorization')
    logger.debug(f'Authorization Header: {auth_header}')
    if not auth_header:
        raise HTTPException(status_code=401, detail="missing_authorization_header")

    token = auth_header.split(' ')[1]
    user_info = tokens.get(token, {}).get("user", None)
    if user_info:
        return JSONResponse(content=user_info)
    else:
        raise HTTPException(status_code=401, detail="User not authorized")

# Mock getaddrinfo Endpoint
@app.get('/getaddrinfo')
async def getaddrinfo(hostname: Optional[str] = "localhost"):
    response = {
        "hostname": hostname,
        "address": "127.0.0.1"
    }
    return JSONResponse(content=response)

# Mock Copilot Token Endpoint for GET requests
@app.get('/copilot_internal/v2/token')
async def copilot_internal_token_get():
    response = {
        "token": "access_token=mock_access_token;token_type=bearer;expires_in=3600;refresh_token=mock_refresh_token",
        "user": {
            "id": "123456",
            "login": "mock_user",
            "name": "Mock User"
        },
        "expires_at": 3600,  # Add this field if it is required
        "refresh_in": 1800  # Add this field if it is required
    }
    return JSONResponse(content=response)

# Mock Copilot Token Endpoint for POST requests
@app.post('/copilot_internal/v2/token')
async def copilot_internal_token_post(request: TokenRequest):
    if not request.client_id or not request.device_code or not request.grant_type:
        raise HTTPException(status_code=400, detail="invalid_request")

    if request.device_code == "mock_device_code":
        response = {
            "token": "access_token=mock_access_token;token_type=bearer;expires_in=3600;refresh_token=mock_refresh_token",
            "user": {
                "id": "123456",
                "login": "mock_user",
                "name": "Mock User"
            },
            "expires_at": 3600,  # Add this field if it is required
            "refresh_in": 1800  # Add this field if it is required
        }
        return JSONResponse(content=response)
    else:
        raise HTTPException(status_code=400, detail="invalid_grant")

# Mock GitHub Enterprise Meta Endpoint
@app.get('/api/v3/meta')
async def github_enterprise_meta():
    response = {
        "verifiable_password_authentication": False,
        "github_services_sha": "mock_sha",
        "installed_version": "mock_version"
    }
    return JSONResponse(content=response)

# Mock Telemetry Endpoint for POST requests
@app.post('/telemetry')
async def telemetry_post(request: Request):
    telemetry_data = await request.json()
    print("Received telemetry data:", telemetry_data)
    return JSONResponse(content={}, status_code=200)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5201)
