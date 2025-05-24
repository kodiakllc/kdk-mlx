from fastapi import FastAPI, Request
import httpx
import os
from urllib.parse import urlparse
from starlette.responses import Response

app = FastAPI()

PROXY_URL = "http://localhost:8080"

# Validate the PROXY_URL
parsed_proxy_url = urlparse(PROXY_URL)
if not parsed_proxy_url.scheme or not parsed_proxy_url.netloc:
    raise ValueError(f"Invalid PROXY_URL: {PROXY_URL}. Please provide a valid proxy URL.")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def forward_request(path: str, request: Request):
    # Update the proxies dictionary to use full URL forms
    proxies = {"http://": PROXY_URL, "https://": PROXY_URL}

    async with httpx.AsyncClient(proxies=proxies) as client:
        # Forward headers and body
        headers = dict(request.headers)
        body = await request.body()

        # Make the proxied request
        response = await client.request(
            method=request.method,
            url=f"{PROXY_URL}/{path}",
            headers=headers,
            content=body,
            timeout=120.0,
        )

        # Return the proxied response
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=dict(response.headers)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
