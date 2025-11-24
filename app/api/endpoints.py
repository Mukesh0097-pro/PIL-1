from fastapi import APIRouter, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from app.core.engine import IndxAI_OS

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Initialize Single Instance of OS
os_instance = IndxAI_OS()

# ... existing imports ...


@router.post("/train-knowledge")
async def train(
    background_tasks: BackgroundTasks,  # Add this parameter
    text_data: str = Form(...),
):
    """
    Industry Grade: Training happens in background.
    Server does not freeze.
    """
    # Send the heavy lifting to the background
    background_tasks.add_task(os_instance.learn_new_data, text_data)

    return {
        "status": "Training started",
        "message": "You can continue chatting while I learn.",
    }


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
    return StreamingResponse(
        os_instance.run_query_generator(query), media_type="application/x-ndjson"
    )


@router.get("/health")
async def health():
    return {"status": "active", "engine": "PIL-VAE Hybrid"}
