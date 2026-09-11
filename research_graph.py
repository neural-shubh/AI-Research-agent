from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from rich.console import Console

console = Console()


class ResearchPlan(BaseModel):
    topic: str
    questions: List[str] = Field(description="Specific questions to answer")
    search_queries: List[str] = Field(description="Web search queries to execute")
    depth: int = Field(default=2, description="Research depth (1-3)")


class ResearchState(BaseModel):
    topic: str
    plan: Optional[ResearchPlan] = None
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    context: str = ""
    report: str = ""
    iteration: int = 0
    max_iterations: int = 3


llm = ChatOllama(model="llama3.1:8b", temperature=0.3, num_predict=512)


PLANNER_PROMPT = """You are a research planner. Given a topic, create a research plan with:
1. 3-5 specific questions to answer
2. 3-5 web search queries to find answers
3. Research depth (1=shallow, 2=medium, 3=deep)

Return ONLY valid JSON matching this schema:
{
  "topic": "string",
  "questions": ["string"],
  "search_queries": ["string"],
  "depth": 1-3
}"""


ANALYZER_PROMPT = """You are a research analyst. Given search results and a question, provide a concise answer with citations.
Format: Answer: [your answer]\nSources: [list of URLs]"""


REPORT_PROMPT = """You are a research report writer. Create a well-structured report with:
1. Executive Summary
2. Key Findings (with citations)
3. Detailed Analysis
4. Conclusions
5. Sources

Use the provided context and research plan."""


def planner_node(state: ResearchState) -> Dict[str, Any]:
    console.print(f"[bold cyan]Planning research for:[/bold cyan] {state.topic}")
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Topic: {state.topic}"),
    ]
    response = llm.invoke(messages)
    try:
        plan_data = json.loads(response.content)
        plan = ResearchPlan(**plan_data)
        console.print(f"[green]Plan created:[/green] {len(plan.questions)} questions, {len(plan.search_queries)} queries")
        return {"plan": plan}
    except Exception as e:
        console.print(f"[red]Planning failed:[/red] {e}")
        return {"plan": ResearchPlan(
            topic=state.topic,
            questions=[f"What is {state.topic}?"],
            search_queries=[state.topic],
            depth=1
        )}


def searcher_node(state: ResearchState) -> Dict[str, Any]:
    from tools import web_searcher
    if not state.plan:
        return {}
    console.print("[bold cyan]Executing searches...[/bold cyan]")
    all_results = []
    for query in state.plan.search_queries:
        results = web_searcher.search_and_fetch(query)
        all_results.extend([r.to_dict() for r in results])
    console.print(f"[green]Found {len(all_results)} results[/green]")
    return {"search_results": all_results}


def indexer_node(state: ResearchState) -> Dict[str, Any]:
    from tools import embedding_store
    console.print("[bold cyan]Indexing results...[/bold cyan]")
    from tools import SearchResult
    search_results = [SearchResult(**r) for r in state.search_results]
    count = embedding_store.add_documents(search_results)
    embedding_store.save()
    console.print(f"[green]Indexed {count} chunks[/green]")
    return {}


def analyzer_node(state: ResearchState) -> Dict[str, Any]:
    if not state.plan:
        return {}
    console.print("[bold cyan]Analyzing findings...[/bold cyan]")
    from tools import retrieve_context
    context_parts = []
    for q in state.plan.questions:
        ctx = retrieve_context.invoke({"query": q, "k": 5})
        context_parts.append(f"Question: {q}\nContext: {ctx}")
    context = "\n\n---\n\n".join(context_parts)
    return {"context": context}


def reporter_node(state: ResearchState) -> Dict[str, Any]:
    console.print("[bold cyan]Generating report...[/bold cyan]")
    messages = [
        SystemMessage(content=REPORT_PROMPT),
        HumanMessage(content=f"Topic: {state.topic}\n\nPlan: {state.plan.model_dump_json() if state.plan else '{}'}\n\nContext:\n{state.context}"),
    ]
    response = llm.invoke(messages)
    return {"report": response.content, "iteration": state.iteration + 1}


def should_continue(state: ResearchState) -> str:
    if state.iteration >= state.max_iterations:
        return "end"
    return "continue"


workflow = StateGraph(ResearchState)
workflow.add_node("planner", planner_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("indexer", indexer_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("reporter", reporter_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "indexer")
workflow.add_edge("indexer", "analyzer")
workflow.add_edge("analyzer", "reporter")
workflow.add_conditional_edges("reporter", should_continue, {"continue": "planner", "end": END})

research_graph = workflow.compile()


def run_deep_research(topic: str, max_iterations: int = 2) -> str:
    initial_state = ResearchState(topic=topic, max_iterations=max_iterations)
    final_state = research_graph.invoke(initial_state)
    return final_state.get("report", "Research failed to produce a report.")