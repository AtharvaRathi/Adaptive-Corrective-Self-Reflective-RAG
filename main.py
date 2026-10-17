"""
main.py
────────
Interactive CLI entry point for the Adaptive Self-Reflective RAG Orchestrator.

Features:
  • Rich terminal output with color-coded node transitions
  • Streams LangGraph events to show the full execution trace
  • Displays retrieval strategy, CRAG loop steps, and self-reflection results
  • Prints the final answer with citation list
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich import box

# ── Bootstrap: load .env before any src imports ────────────────────────────────
load_dotenv()

from src.config import get_settings
from src.graph import get_graph
from src.state import make_initial_state

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # Suppress verbose logs; we print our own rich output
    format="%(levelname)s | %(name)s | %(message)s",
)
# Selectively enable INFO for our own modules so graph trace appears in output
for module in ("src.nodes", "src.components", "src.graph"):
    logging.getLogger(module).setLevel(logging.INFO)

console = Console()

# ── Node display metadata ──────────────────────────────────────────────────────
NODE_STYLES: Dict[str, Dict[str, str]] = {
    "route_question": {
        "label": "ROUTER",
        "icon": "🧭",
        "color": "cyan",
    },
    "retrieve_docs": {
        "label": "RETRIEVAL",
        "icon": "🔍",
        "color": "blue",
    },
    "web_search": {
        "label": "WEB SEARCH",
        "icon": "🌐",
        "color": "yellow",
    },
    "grade_documents": {
        "label": "DOC GRADER",
        "icon": "📋",
        "color": "magenta",
    },
    "transform_query": {
        "label": "REWRITER",
        "icon": "✏️",
        "color": "yellow",
    },
    "generate_answer": {
        "label": "GENERATOR",
        "icon": "💬",
        "color": "green",
    },
    "generate_direct_answer": {
        "label": "DIRECT LLM",
        "icon": "⚡",
        "color": "cyan",
    },
    "check_hallucination_and_faithfulness": {
        "label": "REFLECTION",
        "icon": "🪞",
        "color": "bright_magenta",
    },
}


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]Adaptive Self-Reflective RAG Orchestrator[/bold cyan]\n"
            "[dim]LangGraph · FAISS+BM25 · CRAG · Self-RAG · Structured Evaluation[/dim]\n\n"
            "[dim]Type [bold]'quit'[/bold] or [bold]'exit'[/bold] to stop. "
            "Type [bold]'help'[/bold] for example queries.[/dim]",
            border_style="cyan",
            padding=(1, 4),
        )
    )


def _print_help() -> None:
    table = Table(title="Example Queries", box=box.ROUNDED, border_style="dim")
    table.add_column("Type", style="cyan", no_wrap=True)
    table.add_column("Example Query", style="white")
    table.add_column("Expected Route", style="green", no_wrap=True)

    table.add_row("Semantic", "What are transformer attention mechanisms?", "vectorstore")
    table.add_row("Keyword", "What is BM25 retrieval?", "vectorstore")
    table.add_row("Hybrid", "How does Corrective RAG handle low-relevance docs?", "vectorstore")
    table.add_row("Web", "Latest news about OpenAI GPT-5 release", "web_search")
    table.add_row("Direct", "What is 2 + 2?", "direct_answer")

    console.print(table)


def _print_node_start(node_name: str) -> None:
    style = NODE_STYLES.get(node_name, {"label": node_name.upper(), "icon": "▶", "color": "white"})
    console.print(
        f"  [{style['color']}]{style['icon']} [{style['label']}][/{style['color']}]",
        end=" ",
    )


def _print_event(event: Dict[str, Any]) -> None:
    """Pretty-print a single LangGraph streaming event."""
    for node_name, node_output in event.items():
        if node_name == "__end__":
            continue

        style = NODE_STYLES.get(
            node_name, {"label": node_name.upper(), "icon": "▶", "color": "white"}
        )
        label = style["label"]
        icon = style["icon"]
        color = style["color"]

        # Build status line
        parts = []

        if node_name == "route_question" and "search_strategy" in node_output:
            strategy = node_output["search_strategy"]
            parts.append(f"Strategy → [bold]{strategy}[/bold]")

        elif node_name == "retrieve_docs" and "documents" in node_output:
            n = len(node_output["documents"])
            parts.append(f"Retrieved [bold]{n}[/bold] documents")

        elif node_name == "web_search" and "documents" in node_output:
            n = len(node_output["documents"])
            parts.append(f"Web fetched [bold]{n}[/bold] results")

        elif node_name == "grade_documents":
            n = len(node_output.get("documents", []))
            step = node_output.get("loop_step", "?")
            parts.append(f"[bold]{n}[/bold] relevant docs | Attempt {step}")
            if node_output.get("failure_reason"):
                parts.append(f"[yellow]⚠ Triggering query rewrite[/yellow]")

        elif node_name == "transform_query" and "question" in node_output:
            new_q = node_output["question"]
            parts.append(f"Rewritten → [italic]\"{new_q}\"[/italic]")

        elif node_name == "generate_answer":
            parts.append("Answer generated")

        elif node_name == "check_hallucination_and_faithfulness":
            grounded = node_output.get("is_grounded", False)
            useful = node_output.get("is_useful", False)
            g_str = "[green]✅ Grounded[/green]" if grounded else "[red]❌ Hallucination![/red]"
            u_str = "[green]✅ Useful[/green]" if useful else "[red]❌ Not useful[/red]"
            parts.append(f"{g_str} | {u_str}")

        status = " | ".join(parts) if parts else ""
        console.print(
            f"  [{color}]{icon} [{label}][/{color}]"
            + (f" → {status}" if status else "")
        )


def run_query(question: str) -> None:
    """Run a single query through the RAG graph and print results."""
    console.print()
    console.print(Rule(f"[bold]Query[/bold]: {question}", style="cyan"))

    graph = get_graph()
    initial_state = make_initial_state(question)

    console.print("\n[dim]── Execution Trace ──────────────────────────────────[/dim]")

    final_state = None
    start_time = time.perf_counter()

    try:
        for event in graph.stream(initial_state, stream_mode="updates"):
            _print_event(event)
            # Capture last known state
            for node_output in event.values():
                if isinstance(node_output, dict):
                    if final_state is None:
                        final_state = dict(initial_state)
                    final_state.update(node_output)

    except Exception as exc:
        console.print(f"\n[red]❌ Graph execution error: {exc}[/red]")
        return

    elapsed = time.perf_counter() - start_time
    console.print(f"\n[dim]── Completed in {elapsed:.2f}s ──────────────────────────[/dim]\n")

    if not final_state:
        console.print("[yellow]No output produced.[/yellow]")
        return

    generation = final_state.get("generation", "")
    citations = final_state.get("citations", [])
    strategy = final_state.get("search_strategy", "unknown")
    loop_step = final_state.get("loop_step", 0)

    # ── Print Answer ───────────────────────────────────────────────────────────
    console.print(
        Panel(
            Markdown(generation) if generation else "[dim]No answer generated.[/dim]",
            title=f"[green bold]Answer[/green bold] [dim](strategy: {strategy}, retries: {loop_step})[/dim]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # ── Print Citations ────────────────────────────────────────────────────────
    if citations:
        console.print("\n[bold cyan]📚 Citations:[/bold cyan]")
        for i, cite in enumerate(citations, 1):
            console.print(f"  [dim][{i}][/dim] {cite}")
    else:
        console.print("\n[dim]No citations extracted.[/dim]")


def main() -> None:
    """Main interactive loop."""
    _print_banner()

    settings = get_settings()
    if not settings.groq_api_key:
        console.print(
            "\n[red bold]⚠️  GROQ_API_KEY is not set.[/red bold]\n"
            "Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] and add your Groq API key."
        )
        sys.exit(1)

    console.print(
        f"\n[dim]Model: [bold]{settings.llm_model}[/bold] | "
        f"Embeddings: [bold]{settings.embedding_model}[/bold] | "
        f"Max retries: [bold]{settings.max_retries}[/bold][/dim]\n"
    )

    while True:
        try:
            question = console.input("[bold cyan]❓ Enter your question:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n\n[dim]Goodbye! 👋[/dim]")
            break

        if not question:
            continue

        if question.lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Goodbye! 👋[/dim]")
            break

        if question.lower() in ("help", "h", "?"):
            _print_help()
            continue

        run_query(question)
        console.print()


if __name__ == "__main__":
    main()
