#!/usr/bin/env python3
from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from research_graph import run_deep_research
from tools import embedding_store, web_searcher, retrieve_context

app = typer.Typer(
    name="research-agent",
    help="Local AI Research Agent - No API keys needed",
    rich_markup_mode="rich",
)
console = Console()


def _ensure_loaded():
    embedding_store.load()


@app.command()
def research(
    topic: str = typer.Argument(..., help="Research topic"),
    iterations: int = typer.Option(2, "--iterations", "-i", help="Research iterations (1-3)"),
):
    """Run deep research on a topic."""
    _ensure_loaded()
    console.print(Panel.fit(f"[bold]Deep Research:[/bold] {topic}", border_style="cyan"))
    report = run_deep_research(topic, max_iterations=iterations)
    console.print(Markdown(report))


@app.command()
def quick(
    query: str = typer.Argument(..., help="Search query"),
    k: int = typer.Option(5, "--results", "-k", help="Number of results"),
):
    """Quick web search and summarize."""
    console.print(f"[cyan]Searching:[/cyan] {query}")
    results = web_searcher.search_and_fetch(query)
    for i, r in enumerate(results[:k], 1):
        console.print(f"\n[bold]{i}. {r.title}[/bold]")
        console.print(f"   [dim]{r.url}[/dim]")
        console.print(f"   {r.snippet[:200]}...")
        if r.content:
            console.print(f"   [green]Fetched {len(r.content)} chars[/green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to answer from knowledge base"),
    k: int = typer.Option(5, "--results", "-k", help="Number of context chunks"),
):
    """Ask a question using the local knowledge base."""
    _ensure_loaded()
    console.print(f"[cyan]Question:[/cyan] {question}")
    context = retrieve_context.invoke({"query": question, "k": k})
    console.print(Panel(context, title="Relevant Context", border_style="green"))


@app.command()
def index(
    query: str = typer.Argument(..., help="Topic to search and index"),
):
    """Search a topic and add to knowledge base."""
    from tools import add_to_knowledge_base
    result = add_to_knowledge_base.invoke({"query": query})
    console.print(f"[green]{result}[/green]")


@app.command()
def stats():
    """Show knowledge base statistics."""
    _ensure_loaded()
    console.print(f"[bold]Vectors indexed:[/bold] {embedding_store.index_count}")
    console.print(f"[bold]Metadata entries:[/bold] {len(embedding_store._metadata)}")


@app.command()
def clear():
    """Clear the knowledge base."""
    import shutil
    if embedding_store.persist_dir.exists():
        shutil.rmtree(embedding_store.persist_dir)
    embedding_store._embeddings = []
    embedding_store._metadata = []
    console.print("[green]Knowledge base cleared[/green]")


@app.command()
def interactive():
    """Interactive research session."""
    _ensure_loaded()
    console.print(Panel.fit(
        "[bold]Interactive Research Agent[/bold]\n"
        "Commands: research <topic> | quick <query> | ask <question> | index <topic> | stats | clear | quit",
        border_style="cyan"
    ))
    while True:
        try:
            cmd = Prompt.ask("\n[bold cyan]research>[/bold cyan]")
            if cmd.lower() in ("quit", "exit", "q"):
                break
            parts = cmd.split(maxsplit=1)
            if not parts:
                continue
            action = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if action == "research" and arg:
                report = run_deep_research(arg)
                console.print(Markdown(report))
            elif action == "quick" and arg:
                results = web_searcher.search_and_fetch(arg)
                for r in results[:5]:
                    console.print(f"\n[bold]{r.title}[/bold] - {r.url}")
            elif action == "ask" and arg:
                ctx = retrieve_context.invoke({"query": arg})
                console.print(Panel(ctx, title="Context"))
            elif action == "index" and arg:
                from tools import add_to_knowledge_base
                console.print(add_to_knowledge_base.invoke({"query": arg}))
            elif action == "stats":
                console.print(f"Vectors: {embedding_store.index_count}")
            elif action == "clear":
                import shutil
                if embedding_store.persist_dir.exists():
                    shutil.rmtree(embedding_store.persist_dir)
                embedding_store._embeddings = []
                embedding_store._metadata = []
                console.print("[green]Cleared[/green]")
            else:
                console.print("[yellow]Unknown command[/yellow]")
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
    console.print("[cyan]Goodbye![/cyan]")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive()
    else:
        app()