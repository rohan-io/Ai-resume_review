from fastapi import FastAPI
from routers import resume

app = FastAPI(
    title="Resume Screener API",
    description="Upload a JD, then evaluate resumes against it.",
    version="1.0.0",
)

app.include_router(resume.router)