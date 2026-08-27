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
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .config import Config
from .database import connect, get_schema_info
from .llm import create_llm
from .agent import AgenticSQLAgent
from .auth import AuthDatabase, User, create_jwt_token, decode_jwt_token
from .tenancy import TenantManager

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory sliding-window request rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._records: dict[str, list[float]] = {}

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        timestamps = self._records.get(client_id, [])
        valid_timestamps = [t for t in timestamps if now - t < self.window_seconds]
        if len(valid_timestamps) >= self.max_requests:
            self._records[client_id] = valid_timestamps
            return False
        valid_timestamps.append(now)
        self._records[client_id] = valid_timestamps
        return True


# Module-level state (initialized during lifespan)
_agent: Optional[AgenticSQLAgent] = None
_db = None
_config: Optional[Config] = None
_auth_db = AuthDatabase()
_tenant_manager = TenantManager(_auth_db)
_rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


# --- Request / Response models ---


class ConnectRequest(BaseModel):
    database_url: Optional[str] = None
    db_type: Optional[str] = "mssql"
    db_server: Optional[str] = "127.0.0.1"
    db_name: Optional[str] = ""
    db_user: Optional[str] = ""
    db_password: Optional[str] = ""
    db_driver: Optional[str] = "ODBC+Driver+17+for+SQL+Server"


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: Optional[str] = "user"


class LoginRequest(BaseModel):
    username_or_email: str
    password: str


class AddConnectionRequest(BaseModel):
    name: str
    db_type: str
    db_uri: str
    db_server: Optional[str] = ""
    db_name: Optional[str] = ""
    is_default: Optional[bool] = False


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""
    message: str
    connection_id: Optional[int] = None


class ChatResponse(BaseModel):
    """Response body for the /api/chat endpoint."""
    output: str
    sql: list[str]
    data: Optional[dict] = None
    healed: Optional[bool] = False
    attempts: Optional[int] = 1
    explanation: Optional[str] = ""
    cost: Optional[str] = "LOW"
    profiling_tips: Optional[list[str]] = []
    profiling_warnings: Optional[list[str]] = []


class ExemplarRequest(BaseModel):
    """Request body for the /api/exemplar endpoint."""
    question: str
    sql: str
    category: Optional[str] = "general"


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
            _agent = AgenticSQLAgent(
                llm=llm,
                db=_db,
                verbose=False,
                max_retries=config.max_retries,
                enable_self_healing=config.enable_self_healing,
                enable_schema_pruning=config.enable_schema_pruning,
            )
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

    # Rate Limiting Middleware (Sliding Window: 60 requests/min)
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        # Exclude health check and docs from rate limit
        path = request.url.path
        if path.startswith("/api/health") or path.startswith("/docs") or path.startswith("/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        if not _rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "Request rate limit exceeded (max 60 requests/minute). Please retry shortly."},
            )
        return await call_next(request)

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

    @app.post("/api/connect")
    async def connect_database(req: ConnectRequest):
        """Connect to a new database dynamically."""
        global _agent, _db
        try:
            temp_config = Config(
                google_api_key=_config.google_api_key if _config else "",
                database_url=req.database_url or "",
                db_type=req.db_type or "mssql",
                db_server=req.db_server or "127.0.0.1",
                db_name=req.db_name or "",
                db_user=req.db_user or "",
                db_password=req.db_password or "",
                db_driver=req.db_driver or "ODBC+Driver+17+for+SQL+Server",
            )
            new_db = connect(temp_config)
            llm = create_llm(temp_config)
            new_agent = AgenticSQLAgent(
                llm=llm,
                db=new_db,
                verbose=False,
                max_retries=temp_config.max_retries,
                enable_self_healing=temp_config.enable_self_healing,
                enable_schema_pruning=temp_config.enable_schema_pruning,
            )
            _db = new_db
            _agent = new_agent
            schema = get_schema_info(_db)
            return {
                "status": "connected",
                "database": req.db_name or "Custom Database",
                "dialect": getattr(_db, "dialect", req.db_type),
                "tables_count": len(schema),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to connect to database: {e}")

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
            healed=response.get("healed", False),
            attempts=response.get("attempts", 1),
            explanation=response.get("explanation", ""),
            cost=response.get("cost", "LOW"),
            profiling_tips=response.get("profiling_tips", []),
            profiling_warnings=response.get("profiling_warnings", []),
        )

    @app.post("/api/exemplar")
    async def add_exemplar(request: ExemplarRequest):
        """Save a validated (question, SQL) pair to the few-shot learning store."""
        if not _agent:
            raise HTTPException(status_code=503, detail="Agent not initialized.")

        if not request.question.strip() or not request.sql.strip():
            raise HTTPException(status_code=400, detail="Question and SQL must not be empty.")

        _agent.add_golden_example(
            question=request.question,
            sql=request.sql,
            category=request.category or "general",
        )
        return {"status": "success", "message": "Golden exemplar saved successfully."}

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

    # --- Phase 4c: Auth & Multi-Tenancy Endpoints ---

    def get_current_user(authorization: Optional[str] = Header(None)) -> User:
        """Dependency: Extract and verify JWT Bearer token from request headers."""
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Bearer authorization header.")

        token = authorization.split("Bearer ")[1].strip()
        payload = decode_jwt_token(token)
        if not payload or "sub" not in payload:
            raise HTTPException(status_code=401, detail="Invalid or expired JWT token.")

        user = _auth_db.get_user_by_id(payload["sub"])
        if not user:
            raise HTTPException(status_code=401, detail="Authenticated user account not found.")

        return user

    @app.post("/api/auth/register")
    async def register(req: RegisterRequest):
        """Register a new user account."""
        try:
            user = _auth_db.register_user(req.username, req.email, req.password, role=req.role or "user")
            token = create_jwt_token({"sub": user.id, "username": user.username, "role": user.role})
            return {
                "status": "success",
                "token": token,
                "user": {"id": user.id, "username": user.username, "email": user.email, "role": user.role},
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/api/auth/login")
    async def login(req: LoginRequest):
        """Authenticate user and issue a JWT token."""
        user = _auth_db.authenticate_user(req.username_or_email, req.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid username/email or password.")

        token = create_jwt_token({"sub": user.id, "username": user.username, "role": user.role})
        return {
            "status": "success",
            "token": token,
            "user": {"id": user.id, "username": user.username, "email": user.email, "role": user.role},
        }

    @app.get("/api/auth/me")
    async def get_me(user: User = Depends(get_current_user)):
        """Retrieve current authenticated user profile."""
        return {"user": {"id": user.id, "username": user.username, "email": user.email, "role": user.role}}

    @app.get("/api/tenants/connections")
    async def get_connections(user: User = Depends(get_current_user)):
        """List all database connections registered by the current user."""
        conns = _tenant_manager.get_user_connections(user.id)
        from dataclasses import asdict
        return {"connections": [asdict(c) for c in conns]}

    @app.post("/api/tenants/connections")
    async def add_connection(req: AddConnectionRequest, user: User = Depends(get_current_user)):
        """Register a new tenant database connection for the current user."""
        conn = _tenant_manager.register_connection(
            user_id=user.id,
            name=req.name,
            db_type=req.db_type,
            db_uri=req.db_uri,
            db_server=req.db_server or "",
            db_name=req.db_name or "",
            is_default=req.is_default or False,
        )
        from dataclasses import asdict
        return {"status": "success", "connection": asdict(conn)}

    @app.get("/api/tenants/history")
    async def get_user_history(user: User = Depends(get_current_user)):
        """Get the current user's isolated query history."""
        records = _auth_db.get_user_query_history(user.id)
        from dataclasses import asdict
        return {"history": [asdict(r) for r in records]}

    @app.get("/api/admin/stats")
    async def get_admin_stats(user: User = Depends(get_current_user)):
        """Admin-only: Aggregate system usage metrics, active tenants, and guardrail statistics."""
        if user.role != "admin":
            raise HTTPException(status_code=403, detail="Administrator privileges required.")
        stats = _auth_db.get_admin_stats()
        return {"status": "success", "stats": stats}

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
        WebSocket endpoint for real-time token-streaming chat.

        Sends structured JSON messages:
        - {"type": "status", "message": "..."}
        - {"type": "sql", "sql": [...]}
        - {"type": "token", "chunk": "..."}
        - {"type": "done", "output": "...", "data": ...}
        """
        await websocket.accept()
        try:
            while True:
                data = await websocket.receive_text()

                if not _agent:
                    await websocket.send_json({"type": "error", "error": "Agent not initialized."})
                    continue

                if not data.strip():
                    await websocket.send_json({"type": "error", "error": "Message cannot be empty."})
                    continue

                await websocket.send_json({"type": "status", "message": "Analyzing question & generating SQL..."})

                response = _agent.chat(data)

                # Stream SQL query if generated
                if response.get("sql"):
                    await websocket.send_json({"type": "sql", "sql": response["sql"]})

                # Stream output tokens
                output_text = response.get("output", "")
                words = output_text.split(" ")
                for i in range(0, len(words), 3):
                    chunk = " ".join(words[i:i + 3]) + " "
                    await websocket.send_json({"type": "token", "chunk": chunk})

                await websocket.send_json({
                    "type": "done",
                    "output": output_text,
                    "sql": response.get("sql", []),
                    "data": response.get("data"),
                    "healed": response.get("healed", False),
                    "cost": response.get("cost", "LOW"),
                    "explanation": response.get("explanation", ""),
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
