"""ASGI entry point: `uvicorn main:app` from the Backend folder."""

import uvicorn

from neuroscan.app import create_app
from neuroscan.config import get_settings

app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port, proxy_headers=True)
