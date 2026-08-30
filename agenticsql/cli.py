"""
Interactive REPL and CLI interface for AgenticSQL.

Provides:
- An interactive chat loop with rich terminal output
- Slash commands (/help, /schema, /explain, /export, /chart, /history, /clear, /quit)
- Single-query mode via command line arguments
- Server mode launch via --server flag
"""

import sys
import logging
import argparse
from typing import Optional

# pyrefly: ignore [missing-import]

# pyrefly: ignore [missing-import]
from rich.panel import Panel
# pyrefly: ignore [missing-import]
from rich.syntax import Syntax
# pyrefly: ignore [missing-import]
from rich.markdown import Markdown
# pyrefly: ignore [missing-import]
from rich.console import Console

# pyrefly: ignore [missing-import]
from prompt_toolkit import PromptSession
# pyrefly: ignore [missing-import]
from prompt_toolkit.history import InMemoryHistory
# pyrefly: ignore [missing-import]
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from .config import Config
from .database import connect
from .llm import create_llm
from .agent import AgenticSQLAgent
from .visualization import (
    parse_table_from_text,
    save_chart,
    export_to_csv,
    export_to_json,
)

logger = logging.getLogger(__name__)
console = Console()


HELP_TEXT = """
[bold cyan]Available Commands:[/bold cyan]

  [green]/help[/green]          Show this help message
  [green]/schema[/green]        Display database schema (tables & columns)
  [green]/explain[/green]       Show the SQL from the last query with breakdown
  [green]/save[/green]          Bookmark last query as a golden exemplar (Phase 4c)
  [green]/export csv[/green]    Export last results to CSV
  [green]/export json[/green]   Export last results to JSON
  [green]/chart[/green]         Generate a chart from the last results
  [green]/history[/green]       Show conversation history
  [green]/clear[/green]         Clear conversation history
  [green]/quit[/green]          Exit AgenticSQL

[dim]Or just type a natural language question about your data![/dim]
"""

BANNER = r"""[bold blue]
    _                    _   _      ____   ___  _
   / \   __ _  ___ _ __ | |_(_) ___/ ___| / _ \| |
  / _ \ / _` |/ _ \ '_ \| __| |/ __\___ \| | | | |
 / ___ \ (_| |  __/ | | | |_| | (__ ___) | |_| | |___
/_/   \_\__, |\___|_| |_|\__|_|\___|____/ \___\_\_____|
        |___/
[/bold blue]"""


def run_repl(config: Config) -> None:
    """
    Start the interactive rich terminal REPL (Read-Eval-Print Loop).
    Handles user chat inputs, slash commands (/help, /schema, /export, /chart), and formatted table outputs.
    """
    console.print(BANNER)
    console.print(
        Panel(
            f"[bold]Connected to:[/bold] [cyan]{config.db_name}[/cyan] "
            f"@ [cyan]{config.db_server}[/cyan]\n"
            f"[bold]LLM:[/bold] [cyan]{config.llm_model}[/cyan]\n"
            f"[dim]Type /help for commands or ask a question about your data.[/dim]",
            title="[bold green]AgenticSQL v1.0.0[/bold green]",
            border_style="green",
        )
    )

    # Initialize components
    try:
        with console.status("[bold cyan]Connecting to database...[/bold cyan]", spinner="dots"):
            db = connect(config)
        console.print("[green]✓ Database connected.[/green]")

        with console.status("[bold cyan]Initializing LLM...[/bold cyan]", spinner="dots"):
            llm = create_llm(config)
        console.print("[green]✓ LLM ready.[/green]\n")

        agent = AgenticSQLAgent(
            llm=llm,
            db=db,
            verbose=False,
            max_retries=config.max_retries,
            enable_self_healing=config.enable_self_healing,
            enable_schema_pruning=config.enable_schema_pruning,
        )

    except Exception as e:
        console.print(f"\n[bold red]Startup Error:[/bold red] {e}")
        sys.exit(1)

    # Track last response for /explain, /export, /chart, /save
    last_response: dict = {}
    last_input_text: str = ""

    # Prompt session with history and auto-suggestions
    session = PromptSession(
        history=InMemoryHistory(),
        auto_suggest=AutoSuggestFromHistory(),
    )

    while True:
        try:
            user_input = session.prompt("\n🔷 agenticsql> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # --- Handle slash commands ---
        if user_input.startswith("/"):
            parts = user_input.lower().split()
            cmd = parts[0]

            if cmd in ("/quit", "/exit"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                console.print(HELP_TEXT)

            elif cmd == "/schema":
                _show_schema(agent)

            elif cmd == "/explain":
                _show_last_sql(agent, last_response)

            elif cmd == "/save":
                _save_exemplar(agent, last_input_text, last_response)

            elif cmd == "/export":
                fmt = parts[1] if len(parts) > 1 else "csv"
                _export_results(agent, last_response, fmt)

            elif cmd == "/chart":
                _generate_chart(agent, last_response)

            elif cmd == "/history":
                _show_history(agent)

            elif cmd == "/clear":
                agent.clear_history()
                last_response = {}
                last_input_text = ""
                console.print("[green]✓ Conversation history cleared.[/green]")

            else:
                console.print(
                    f"[yellow]Unknown command: {cmd}. Type /help for options.[/yellow]"
                )

            continue

        # --- Regular query ---
        last_input_text = user_input
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            response = agent.chat(user_input)

        last_response = response

        # Display the agent's response
        console.print()
        console.print(
            Panel(
                Markdown(response["output"]),
                title="[bold green]Agent Response[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        # Display self-healing notification if query was repaired
        if response.get("healed"):
            console.print(
                f"[bold cyan]🔄 Query was automatically self-healed in {response.get('attempts', 2)} attempts.[/bold cyan]"
            )

        # Display query performance profile (Phase 4c)
        cost = response.get("cost", "LOW")
        cost_color = "green" if cost == "LOW" else ("yellow" if cost == "MEDIUM" else "red")
        console.print(f"[{cost_color}]⚡ Query Performance Rating: {cost}[/{cost_color}]")

        tips = response.get("profiling_tips", [])
        if tips:
            for tip in tips:
                console.print(f"  [dim]💡 Tip: {tip}[/dim]")

        # Show generated SQL if available
        if response.get("sql"):
            for sql in response["sql"]:
                console.print(
                    Panel(
                        Syntax(sql, "sql", theme="monokai", word_wrap=True),
                        title="[bold yellow]Generated SQL[/bold yellow]",
                        border_style="yellow",
                    )
                )


# --- Slash command handlers ---


def _show_schema(agent: AgenticSQLAgent) -> None:
    """Display the database schema."""
    with console.status("[bold cyan]Fetching schema...[/bold cyan]", spinner="dots"):
        schema = agent.get_schema()

    console.print(
        Panel(
            Syntax(schema, "sql", theme="monokai", word_wrap=True),
            title="[bold blue]Database Schema[/bold blue]",
            border_style="blue",
        )
    )


def _show_last_sql(agent: AgenticSQLAgent, last_response: Optional[dict] = None) -> None:
    """Show the SQL from the last query along with educational breakdown."""
    if agent.last_sql:
        console.print(
            Panel(
                Syntax(agent.last_sql, "sql", theme="monokai", word_wrap=True),
                title="[bold yellow]Last Generated SQL[/bold yellow]",
                border_style="yellow",
            )
        )
        explanation = (last_response or {}).get("explanation")
        if not explanation:
            explanation = agent.explain_sql(agent.last_sql)
        if explanation:
            console.print(
                Panel(
                    Markdown(explanation),
                    title="[bold cyan]💡 Query Explanation Breakdown[/bold cyan]",
                    border_style="cyan",
                )
            )
    else:
        console.print("[yellow]No SQL query available yet. Ask a question first.[/yellow]")


def _save_exemplar(agent: AgenticSQLAgent, user_question: str, last_response: dict) -> None:
    """Save the last question and executed SQL as a golden few-shot exemplar (Phase 4c)."""
    sql = agent.last_sql or ((last_response.get("sql") or [])[-1] if last_response.get("sql") else None)
    if not sql or not user_question:
        console.print("[yellow]No active query to save as an exemplar. Ask a question first.[/yellow]")
        return

    agent.add_golden_example(question=user_question, sql=sql)
    console.print(f"[green]✓ Saved golden exemplar for: '{user_question}'[/green]")


def _export_results(agent: AgenticSQLAgent, response: dict, fmt: str) -> None:
    """Export the last results to a file."""
    # Prioritize direct DataFrame from agent
    df = getattr(agent, "last_df", None)
    if df is None and response.get("data"):
        from .visualization import _to_dataframe
        df = _to_dataframe(response["data"])

    data_source = df if df is not None else response.get("output", "")
    if not data_source:
        console.print("[yellow]No results to export. Ask a question first.[/yellow]")
        return

    if fmt == "csv":
        path = export_to_csv(data_source)
    elif fmt == "json":
        path = export_to_json(data_source)
    else:
        console.print(f"[yellow]Unsupported format: {fmt}. Use 'csv' or 'json'.[/yellow]")
        return

    if path:
        console.print(f"[green]✓ Exported to: {path}[/green]")
    else:
        console.print("[red]Export failed. Check logs for details.[/red]")


def _generate_chart(agent: AgenticSQLAgent, response: dict) -> None:
    """Generate a chart from the last response."""
    # Prioritize direct DataFrame from agent
    df = getattr(agent, "last_df", None)
    if df is None and response.get("data"):
        from .visualization import _to_dataframe
        df = _to_dataframe(response["data"])

    data_source = df if df is not None else response.get("output", "")
    if not data_source:
        console.print("[yellow]No results to chart. Ask a question first.[/yellow]")
        return

    with console.status("[bold cyan]Generating chart...[/bold cyan]", spinner="dots"):
        path = save_chart(data_source)

    if path:
        console.print(f"[green]✓ Chart saved to: {path}[/green]")
    else:
        console.print("[yellow]Could not generate chart from the data.[/yellow]")


def _show_history(agent: AgenticSQLAgent) -> None:
    """Show conversation history."""
    history = agent.get_history()
    if not history:
        console.print("[yellow]No conversation history yet.[/yellow]")
        return

    for i, entry in enumerate(history):
        if entry["role"] == "user":
            console.print(f"  [bold cyan]You:[/bold cyan] {entry['content']}")
        else:
            # Truncate long assistant responses for readability
            content = entry["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            console.print(f"  [bold green]Agent:[/bold green] {content}")
    console.print()


# --- CLI argument parsing ---


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="agenticsql",
        description="AgenticSQL — Chat with your database using natural language.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Run a single query and exit (non-interactive mode).",
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server instead of the CLI.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Start the Streamlit Web Dashboard UI.",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Server host (default: from .env or 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: from .env or 8000).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose agent output (shows intermediate reasoning).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for AgenticSQL."""
    args = parse_args()

    # --- UI mode ---
    if args.ui:
        import subprocess
        from pathlib import Path
        ui_script = Path(__file__).resolve().parent / "ui.py"
        cmd = [sys.executable, "-m", "streamlit", "run", str(ui_script)]
        if args.port:
            cmd.extend(["--server.port", str(args.port)])
        if args.host:
            cmd.extend(["--server.address", str(args.host)])
        sys.exit(subprocess.call(cmd))

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Load and validate config
    try:
        config = Config.from_env()
    except ValueError as e:
        console.print(f"[bold red]Configuration Error:[/bold red]\n{e}")
        sys.exit(1)

    # --- Server mode ---
    if args.server:
        from .server import start_server

        host = args.host or config.server_host
        port = args.port or config.server_port
        start_server(config, host=host, port=port)
        return

    # --- Single query mode ---
    if args.query:
        try:
            db = connect(config)
            llm = create_llm(config)
            agent = AgenticSQLAgent(llm=llm, db=db, verbose=args.verbose)
            response = agent.chat(args.query)
            console.print(response["output"])
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
        return

    # --- Interactive REPL mode (default) ---
    run_repl(config)
