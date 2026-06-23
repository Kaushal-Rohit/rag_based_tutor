"""
Report Endpoint
================
``GET /report`` — renders the data-driven report page.
``GET /api/v1/report/data`` — returns raw report data as JSON.

Uses raw Jinja2 rendering to work around a Python 3.14 compatibility
issue with Starlette's TemplateResponse + Jinja2's LRU cache.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from jinja2 import Environment, FileSystemLoader

from app.services.report import generate_report_data

router = APIRouter()
_env = Environment(loader=FileSystemLoader("app/templates"), autoescape=True)


@router.get(
    "/report",
    response_class=HTMLResponse,
    summary="System Report",
    include_in_schema=False,
)
async def report_page(request: Request):
    """Render the full system report with live metrics."""
    data = generate_report_data(request.app.state)
    template = _env.get_template("report.html")
    html = template.render(request=request, data=data)
    return HTMLResponse(content=html)


@router.get(
    "/api/v1/report/data",
    summary="Report data (JSON)",
    description="Returns the raw report data as JSON for programmatic export.",
)
async def report_data(request: Request):
    """Return raw report data for export or external consumption."""
    data = generate_report_data(request.app.state)
    return JSONResponse(content=data)
