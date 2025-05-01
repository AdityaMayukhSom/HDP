from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
web_router = APIRouter(prefix="")


@web_router.get("/", response_class=HTMLResponse)
async def summarisation_application(request: Request):
    return templates.TemplateResponse(
        request=request, name="application.html", context={}
    )
