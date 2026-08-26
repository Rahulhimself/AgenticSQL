"""
FastAPI server for AgenticSQL.

Provides a REST API and WebSocket endpoint for querying the database
via the AI agent. Designed for integration with web frontends.

Endpoints:
    GET  /api/health    — Health check
    POST /api/chat      — Send a question, get an answer
    GET  /api/history   — Get conversation history
    POST /api/clear     — Clear conversation history
    GET  /api/schema    — Get database schema as JSON
    WS   /ws/chat       — WebSocket for streaming chat
"""

import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import Config
from .database import connect, get_schema_info
from .llm import create_llm
from .agent import AgenticSQLAgent

logger = logging.getLogger(__name__)

# Module-level state (initialized during lifespan)
_agent: Optional[AgenticSQLAgent] = None
_db = None
_config: Optional[Config] = None


# --- Request / Response models ---


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""
    message: str


class ChatResponse(BaseModel):
    """Response body for the /api/chat endpoint."""
    output: str
    sql: list[str]
    data: Optional[dict] = None


class ChartRequest(BaseModel):
    """Request body for the /api/chart endpoint."""
    chart_type: str = "auto"
    title: Optional[str] = None


# --- App factory ---


def create_app(config: Config) -> FastAPI:
    """Create and configure the FastAPI application."""
    global _config
    _config = config

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize database and agent on startup, cleanup on shutdown."""
        global _agent, _db
        logger.info("Starting AgenticSQL API server...")

        try:
            _db = connect(config)
            llm = create_llm(config)
            _agent = AgenticSQLAgent(llm=llm, db=_db, verbose=False)
            logger.info("AgenticSQL API server ready.")
        except Exception as e:
            logger.error("Failed to start server: %s", e)
            raise

        yield

        logger.info("Shutting down AgenticSQL API server.")

    app = FastAPI(
        title="AgenticSQL API",
        description="Chat with your database using natural language.",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — allow all origins for development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Endpoints ---

    @app.get("/api/health")
    async def health_check():
        """Health check — returns server status and configuration info."""
        return {
            "status": "healthy",
            "database": config.db_name,
            "server": config.db_server,
            "dialect": getattr(_db, "dialect", config.db_type) if _db else config.db_type,
            "model": config.llm_model,
        }

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """Send a natural language question and get an answer with generated SQL and structured tabular data."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized.")

        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        response = _agent.chat(request.message)
        return ChatResponse(
            output=response["output"],
            sql=response.get("sql", []),
            data=response.get("data"),
        )

    @app.post("/api/chart")
    async def generate_chart(request: ChartRequest = ChartRequest()):
        """Generate a chart from the last executed SQL query results."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized.")

        if _agent.last_df is None or _agent.last_df.empty:
            raise HTTPException(status_code=400, detail="No query result data available to chart. Execute a query first.")

        from .visualization import save_chart_from_dataframe

        chart_path = save_chart_from_dataframe(
            _agent.last_df,
            chart_type=request.chart_type,
            title=request.title or "",
        )
        if not chart_path:
            raise HTTPException(status_code=422, detail="Could not generate chart from the available data.")

        return {"status": "success", "chart_path": chart_path}

    @app.get("/api/history")
    async def get_history():
        """Get the current conversation history."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized.")
        return {"history": _agent.get_history()}

    @app.post("/api/clear")
    async def clear_history():
        """Clear the conversation history."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized.")
        _agent.clear_history()
        return {"status": "cleared"}

    @app.get("/api/schema")
    async def get_schema():
        """Get the database schema (tables and column definitions)."""
        if not _db:
            raise HTTPException(status_code=503, detail="Database not connected.")
        schema = get_schema_info(_db)
        return {"schema": schema}

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket):
        """
        WebSocket endpoint for real-time chat.

        Send a text message, receive a JSON response with 'output' and 'sql'.
        """
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()

                if not _agent:
                    await websocket.send_json({"error": "Agent not initialized."})
                    continue

                if not data.strip():
                    await websocket.send_json({"error": "Message cannot be empty."})
                    continue

                response = _agent.chat(data)
                await websocket.send_json({
                    "output": response["output"],
                    "sql": response.get("sql", []),
                })

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected.")
        except Exception as e:
            logger.error("WebSocket error: %s", e)

    return app


def start_server(config: Config, host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI server with uvicorn."""
    import uvicorn

    app = create_app(config)
    logger.info("Starting server at http://%s:%d", host, port)
    logger.info("API docs available at http://%s:%d/docs", host, port)
    uvicorn.run(app, host=host, port=port)
