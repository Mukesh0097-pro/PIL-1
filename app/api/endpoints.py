from fastapi import APIRouter, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from app.core.engine import IndxAI_OS

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Initialize Single Instance of OS
os_instance = IndxAI_OS()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the Investor Dashboard"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "history": os_instance.memory.history}
    )


@router.post("/chat")
async def chat(request: Request, query: str = Form(...), mode: str = Form("assistant")):
    """API Endpoint for interaction"""
    os_instance.mode = mode
    response, latency = os_instance.run_query(query)

    return JSONResponse(
        {"response": response, "latency": f"{latency:.2f}ms", "mode": mode}
    )


@router.get("/health")
async def health():
    return {"status": "active", "engine": "PIL-VAE Hybrid"}
