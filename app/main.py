from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import router

app = FastAPI(
    title="indxai Platform",
    description="Gradient-Free Edge AI Operating System",
    version="1.0.0",
)

# Mount Routes
app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
