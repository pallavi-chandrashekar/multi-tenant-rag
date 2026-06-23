import logging
import os
import shutil

from fastapi import FastAPI, Header, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

# Import database utilities
from backend.database import engine, get_db, Base
from backend.models import Document
from backend.api import routes  # Ensure this file exists!
from backend.api import auth_routes
from backend.observability.tracing import init_tracing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Database Tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Enterprise RAG Platform")

# Initialise OpenTelemetry tracing (no-op unless OTEL_ENABLED).
init_tracing(app)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes (search/ingest/chats/documents/eval) and auth.
app.include_router(routes.router)
app.include_router(auth_routes.router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}