#!/usr/bin/env python3
from __future__ import annotations
import sys, typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from research_graph import run_deep_research
from tools import (
    embedding_store, web_searcher, retrieve_context, 
    quick_search_with_answer, ingest_local_documents,
    add_to_knowledge_base
)

app = typer.Typer(name="research-agent", help="Local AI Research Agent - No API keys needed", rich_markup_mode="rich")
console = Console()

def _ensure_loaded(): embedding_store.load()

@app.command()
def research(topic: str, iterations: int = typer.Option(2, "--iterations", "-i")):
    _ensure_loaded()
    console.print(Panel.fit(f"[bold]Deep Research:[/bold] {topic}", border_style="cyan"))
    console.print(Markdown(run_deep_research(topic, max_iterations=iterations)))

@app.command()
def quick(query: str, k: int = typer.Option(5, "--results", "-k"), answer: bool = typer.Option(True, "--answer/--no-answer", help="Synthesize answer with LLM")):
    """Quick web search with optional LLM-synthesized answer."""
    if answer:
        console.print(f"[cyan]Searching & synthesizing:[/cyan] {query}")
        result = quick_search_with_answer(query, k=k)
        console.print(Panel(result["answer"], title="Answer", border_style="green"))
        if result["sources"]:
            console.print("\n[bold]Sources:[/bold]")
            for i, s in enumerate(result["sources"], 1):
                icon = {"wiki": "[W]", "arxiv": "[A]", "semantic_scholar": "[S]", "github": "[G]", "stackoverflow": "[SO]", "local": "[L]"}.get(s.get('source_type', ''), '[D]')
                console.print(f"  [{i}] {icon} {s['title']} - {s['url']}")
    else:
        console.print(f"[cyan]Searching:[/cyan] {query}")
        for i, r in enumerate(web_searcher.search_and_fetch(query)[:k], 1):
            console.print(f"\n[bold]{i}. {r.title}[/bold]\n   [dim]{r.url}[/dim]\n   {r.snippet[:200]}...")
            if r.content: console.print(f"   [green]Fetched {len(r.content)} chars[/green]")

@app.command()
def ask(question: str, k: int = typer.Option(5, "--results", "-k")):
    _ensure_loaded()
    console.print(f"[cyan]Question:[/cyan] {question}")
    console.print(Panel(retrieve_context.invoke({"query": question, "k": k}), title="Relevant Context", border_style="green"))

@app.command()
def index(query: str):
    console.print(f"[green]{add_to_knowledge_base.invoke({'query': query})}[/green]")

@app.command()
def ingest(path: str, recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R")):
    """Ingest local documents (PDF, MD, .py, .ipynb) from a directory."""
    _ensure_loaded()
    console.print(f"[green]{ingest_local_documents.invoke({'path': path, 'recursive': recursive})}[/green]")

@app.command()
def stats():
    _ensure_loaded()
    console.print(f"[bold]Vectors indexed:[/bold] {embedding_store.index_count}\n[bold]Metadata entries:[/bold] {len(embedding_store._metadata)}")

@app.command()
def clear():
    import shutil
    if embedding_store.persist_dir.exists(): shutil.rmtree(embedding_store.persist_dir)
    embedding_store._embeddings = []; embedding_store._metadata = []
    console.print("[green]Knowledge base cleared[/green]")

@app.command()
def interactive():
    _ensure_loaded()
    console.print(Panel.fit("[bold]Interactive Research Agent[/bold]\nCommands: research <topic> | quick <query> | ask <question> | index <topic> | ingest <path> | stats | clear | quit", border_style="cyan"))
    while True:
        try:
            cmd = Prompt.ask("\n[bold cyan]research>[/bold cyan]")
            if cmd.lower() in ("quit", "exit", "q"): break
            parts = cmd.split(maxsplit=1)
            if not parts: continue
            action, arg = parts[0].lower(), parts[1] if len(parts) > 1 else ""
            if action == "research" and arg: console.print(Markdown(run_deep_research(arg)))
            elif action == "quick" and arg:
                result = quick_search_with_answer(arg)
                console.print(Panel(result["answer"], title="Answer", border_style="green"))
            elif action == "ask" and arg: console.print(Panel(retrieve_context.invoke({"query": arg}), title="Context"))
            elif action == "index" and arg: console.print(add_to_knowledge_base.invoke({"query": arg}))
            elif action == "ingest" and arg: console.print(ingest_local_documents(arg))
            elif action == "stats": console.print(f"Vectors: {embedding_store.index_count}")
            elif action == "clear":
                import shutil
                if embedding_store.persist_dir.exists(): shutil.rmtree(embedding_store.persist_dir)
                embedding_store._embeddings = []; embedding_store._metadata = []
                console.print("[green]Cleared[/green]")
            else: console.print("[yellow]Unknown command[/yellow]")
        except KeyboardInterrupt: break
        except Exception as e: console.print(f"[red]Error:[/red] {e}")
    console.print("[cyan]Goodbye![/cyan]")

if __name__ == "__main__":
    if len(sys.argv) == 1: interactive()
    else: app()