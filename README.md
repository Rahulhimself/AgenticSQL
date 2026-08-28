# AgenticSQL

A conversational AI application that lets you chat with your SQL database (MSSQL, PostgreSQL, MySQL, SQLite) using natural language. Built with LangChain, Groq API (Llama 3.3 70B), Google Gemini, and FastAPI/Streamlit.

> Ask questions in plain English → AgenticSQL writes the SQL → executes it → explains the results.

---

## What It Does

AgenticSQL acts as a bridge between normal human questions and complex database queries. Instead of writing manual SQL, you can ask the agent questions about your data (like "Give me a summary of the row counts in all tables"), and it will automatically inspect the database schema, write the correct SQL query, execute it, and return the answer in plain English.

## Key Features

* **Natural Language to SQL:** Uses a LangChain ReAct agent with native tool-calling to autonomously query databases.
* **Powered by Groq & Multi-Provider LLMs:** Ultra-fast sub-second reasoning with Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) with support for Gemini and OpenAI.
* **Multi-Dialect Support:** Connects directly to Microsoft SQL Server/Azure SQL, PostgreSQL, MySQL/MariaDB, and SQLite.
* **Secure by Design:** Keeps all credentials safely out of the codebase using environment variables.
* **Conversation Memory:** Multi-turn context — ask follow-up questions and the agent remembers prior answers.
* **SQL Safety Guardrails:** AST-based inspection blocks destructive queries (DROP, DELETE, ALTER, INSERT, UPDATE) with a persistent audit log.
* **Self-Healing Queries:** Autonomous reflection and iterative query error recovery.
* **AST Query Profiler & Few-Shot RAG:** Identifies query anti-patterns, recommends indexes, and injects golden exemplars into prompts.
* **Interactive UI & REST API:** Streamlit AI Studio and FastAPI backend with WebSocket streaming.

---

## Architecture

```
AgenticSQL/
├── agenticsql/              # Core package
│   ├── config.py            # Environment-based configuration with validation
│   ├── database.py          # Database connection with error handling
│   ├── llm.py               # LLM initialization (Google Gemini)
│   ├── agent.py             # SQL Agent with memory & guardrails
│   ├── guardrails.py        # SQL validation & audit logging
│   ├── visualization.py     # Chart generation & data export
│   ├── cli.py               # Interactive REPL & CLI
│   └── server.py            # FastAPI REST API & WebSocket
├── tests/                   # Unit tests
├── main.py                  # Entry point
├── .env                     # Your credentials (not committed)
├── .env.example             # Template for new contributors
├── requirements.txt
└── pyproject.toml
```

---

## Prerequisites

* **Python 3.10+**
* A local **Microsoft SQL Server** instance with SQL Server Authentication enabled
* **ODBC Driver 17 for SQL Server** installed on your machine
* A **Google AI Studio API key** ([get one here](https://aistudio.google.com/))

---

## Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Rahulhimself/AgenticSQL.git
cd AgenticSQL
```

**2. Create a virtual environment** (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Configure your environment variables**

Copy the template and fill in your credentials:
```bash
copy .env.example .env    # Windows
# cp .env.example .env    # macOS/Linux
```

Edit `.env` with your values:
```env
GOOGLE_API_KEY="your_google_api_key"
DB_USER="your_sql_server_username"
DB_PASSWORD="your_database_password"
DB_SERVER="127.0.0.1"
DB_NAME="your_database_name"
```

> ⚠️ **Never commit `.env` to version control.** It's already in `.gitignore`.

---

## Usage

### Interactive REPL (default)

```bash
python main.py
```

This launches a rich terminal interface:

```
🔷 agenticsql> Show me all tables and their row counts

┌─ Agent Response ──────────────────────────┐
│ Here are the tables in your database...   │
└───────────────────────────────────────────┘
┌─ Generated SQL ───────────────────────────┐
│ SELECT t.name, p.rows FROM sys.tables t   │
│ JOIN sys.partitions p ON ...              │
└───────────────────────────────────────────┘

🔷 agenticsql> How many of those have more than 100 rows?
(Agent remembers the previous context)
```

### Slash Commands

| Command | Description |
|---|---|
| `/help` | Show available commands |
| `/schema` | Display database tables and columns |
| `/explain` | Show the SQL from the last query |
| `/export csv` | Export last results to CSV |
| `/export json` | Export last results to JSON |
| `/chart` | Generate a chart from last results |
| `/history` | Show conversation history |
| `/clear` | Clear conversation history |
| `/quit` | Exit the application |

### Single Query Mode

```bash
python main.py "How many customers are in the database?"
```

### Web Dashboard UI Mode (Phase 4a)

```bash
python main.py --ui
```

Launches an interactive dark-themed web dashboard with:
* **Conversational AI Studio:** Multi-turn chat with real-time SQL syntax accordion.
* **Auto-Visualization Studio:** Interactive Plotly charts (Bar, Line, Area, Pie, Scatter).
* **1-Click Data Export:** Direct export of query results to CSV and JSON.
* **Live Schema Explorer:** Interactive table inspector, column metadata, and data preview.
* **Safety & Audit Monitor:** Real-time query audit logs and guardrail rejection telemetry.

### API Server Mode

```bash
python main.py --server
```

Starts a FastAPI server at `http://localhost:8000` with auto-generated docs at `/docs`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/chat` | Send a question, get an answer |
| `GET` | `/api/history` | Get conversation history |
| `POST` | `/api/clear` | Clear conversation history |
| `GET` | `/api/schema` | Get database schema |
| `WS` | `/ws/chat` | WebSocket for streaming chat |

### Verbose Mode

Add `-v` to see the agent's step-by-step reasoning:
```bash
python main.py -v
python main.py --server -v
```

---

## Safety & Guardrails

AgenticSQL is designed for **read-only** database access:

* ✅ `SELECT` queries are allowed
* ⛔ `DROP`, `DELETE`, `ALTER`, `INSERT`, `UPDATE`, `EXEC`, `CREATE`, `GRANT`, `REVOKE`, `TRUNCATE` are **blocked**
* 📋 Every query attempt is logged to `logs/query_audit.log`

For maximum safety, also configure your SQL Server user with `SELECT`-only permissions.

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Agent Framework | LangChain (ReAct agent) |
| Database | Microsoft SQL Server via SQLAlchemy + PyODBC |
| API Server | FastAPI + Uvicorn |
| Terminal UI | Rich + Prompt Toolkit |
| Visualization | Matplotlib |
| Testing | Pytest |

---

## License

MIT
