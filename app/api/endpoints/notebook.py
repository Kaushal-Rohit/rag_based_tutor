"""
Notebook Endpoint
==================
``GET /notebook`` — serves the NotebookLM-style chat interface.

This is a server-rendered HTML page (Jinja2 template) that uses SSE
streaming from the existing ``/api/v1/query`` endpoint. No separate
frontend build step or Node.js required.

Uses raw Jinja2 rendering to work around a Python 3.14 compatibility
issue with Starlette's TemplateResponse + Jinja2's LRU cache.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

router = APIRouter()
_env = Environment(loader=FileSystemLoader("app/templates"), autoescape=True)


@router.get(
    "/notebook",
    response_class=HTMLResponse,
    summary="Adaptive Notebook UI",
    include_in_schema=False,
)
async def notebook_page(request: Request):
    """Render the notebook chat interface."""
    template = _env.get_template("notebook.html")
    html = template.render(request=request)
    return HTMLResponse(content=html)
