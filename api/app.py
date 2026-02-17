from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from .routes import router
from .dependencies import get_agent, get_conversation_manager
from agent.core import MemoryAgent
from agent.conversation import ConversationManager

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI"""
    # Startup
    logger.info("Starting API server...")
    agent = MemoryAgent()
    await agent.initialize()
    app.state.agent = agent
    yield
    # Shutdown
    logger.info("Shutting down API server...")
    await agent.shutdown()

app = FastAPI(
    title="Agentic Memory API",
    description="API for interacting with the Agentic Memory system",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["memory"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic Memory API",
        "version": "0.1.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}