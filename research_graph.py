from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
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
    structured_report: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 3

llm = ChatOllama(model="phi3:3.8b", temperature=0.3, num_predict=512)

PLANNER_PROMPT = """You are a research planner. Given a topic, create a research plan with:
1. 3-5 specific questions to answer
2. 3-5 web search queries to find answers
3. Research depth (1=shallow, 2=medium, 3=deep)
Return ONLY valid JSON: {"topic":"string","questions":["string"],"search_queries":["string"],"depth":1-3}"""

REPORT_PROMPT = """You are a research report writer. Create a well-structured report with:
1. Executive Summary
2. Key Findings (with numeric citations [1], [2], etc.)
3. Detailed Analysis
4. Conclusions
5. Future Work & Limitations
6. Sources (numbered list matching citations)

Use the provided context and research plan. Cite sources with [1], [2], etc. matching the Sources list."""

def robust_json_parse(content: str, fallback: Optional[Dict] = None) -> Dict:
    """Robustly parse JSON from LLM output with multiple fallback strategies."""
    # Strategy 1: Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first { to last }
    brace_start = content.find('{')
    brace_end = content.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            return json.loads(content[brace_start:brace_end+1])
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Try to fix common issues
    fixed = content
    fixed = re.sub(r',\s*}', '}', fixed)  # Trailing commas
    fixed = re.sub(r',\s*]', ']', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    console.print(f"[red]All JSON parsing strategies failed. Content preview:[/red] {content[:200]}")
    if fallback:
        return fallback
    raise ValueError("Could not parse JSON from LLM output")

def planner_node(state: ResearchState) -> Dict[str, Any]:
    console.print(f"[bold cyan]Planning research for:[/bold cyan] {state.topic}")
    resp = llm.invoke([SystemMessage(content=PLANNER_PROMPT), HumanMessage(content=f"Topic: {state.topic}")])
    
    fallback = {
        "topic": state.topic,
        "questions": [f"What is {state.topic}?", f"How does {state.topic} work?", f"What are applications of {state.topic}?"],
        "search_queries": [state.topic, f"{state.topic} overview", f"{state.topic} applications"],
        "depth": 1
    }
    
    try:
        plan_data = robust_json_parse(resp.content, fallback)
        plan = ResearchPlan(**plan_data)
        console.print(f"[green]Plan created:[/green] {len(plan.questions)} questions, {len(plan.search_queries)} queries")
        return {"plan": plan}
    except Exception as e:
        console.print(f"[red]Planning failed:[/red] {e}")
        return {"plan": ResearchPlan(**fallback)}

def searcher_node(state: ResearchState) -> Dict[str, Any]:
    from tools import web_searcher
    if not state.plan:
        return {}
    console.print("[bold cyan]Executing searches...[/bold cyan]")
    all_results = []
    for query in state.plan.search_queries:
        all_results.extend([r.to_dict() for r in web_searcher.search_and_fetch(query)])
    
    # Also search additional sources
    from sources import search_all_sources
    try:
        additional = search_all_sources(state.topic, max_per_source=2)
        for r in additional:
            all_results.append(r.__dict__)
    except Exception as e:
        console.print(f"[yellow]Additional sources failed:[/yellow] {e}")
    
    console.print(f"[green]Found {len(all_results)} total results[/green]")
    return {"search_results": all_results}

def indexer_node(state: ResearchState) -> Dict[str, Any]:
    from tools import embedding_store, SearchResult
    console.print("[bold cyan]Indexing results...[/bold cyan]")
    count = embedding_store.add_documents([SearchResult(**r) for r in state.search_results])
    embedding_store.save()
    console.print(f"[green]Indexed {count} chunks[/green]")
    return {}

def analyzer_node(state: ResearchState) -> Dict[str, Any]:
    if not state.plan:
        return {}
    console.print("[bold cyan]Analyzing findings...[/bold cyan]")
    from tools import retrieve_context
    ctx = "\n\n---\n\n".join(
        f"Question: {q}\nContext: {retrieve_context.invoke({'query': q, 'k': 5})}"
        for q in state.plan.questions
    )
    return {"context": ctx}

def reporter_node(state: ResearchState) -> Dict[str, Any]:
    console.print("[bold cyan]Generating report...[/bold cyan]")
    
    # Build source list with citation numbers
    source_list = []
    for i, src in enumerate(state.search_results, 1):
        source_list.append(f"[{i}] {src.get('title', 'Unknown')} - {src.get('url', '')}")
    sources_text = "\n".join(source_list)
    
    resp = llm.invoke([
        SystemMessage(content=REPORT_PROMPT),
        HumanMessage(content=f"Topic: {state.topic}\n\nPlan: {state.plan.model_dump_json() if state.plan else '{}'}\n\nSources:\n{sources_text}\n\nContext:\n{state.context}")
    ])
    
    report = resp.content
    
    # Try to structure the report
    structured = {
        "executive_summary": "",
        "key_findings": [],
        "detailed_analysis": "",
        "conclusions": "",
        "future_work": "",
        "sources": []
    }
    
    # Simple extraction from report
    sections = {
        "executive_summary": r"(?:executive summary|summary)[\s:]*([\s\S]*?)(?=\n\s*(?:key findings|detailed analysis|conclusions|$))",
        "key_findings": r"(?:key findings?)[\s:]*([\s\S]*?)(?=\n\s*(?:detailed analysis|conclusions|$))",
        "detailed_analysis": r"(?:detailed analysis)[\s:]*([\s\S]*?)(?=\n\s*(?:conclusions|future work|$))",
        "conclusions": r"(?:conclusions?)[\s:]*([\s\S]*?)(?=\n\s*(?:future work|$))",
        "future_work": r"(?:future work|limitations?)[\s:]*([\s\S]*?)(?=\n\s*(?:sources|$))",
    }
    
    for key, pattern in sections.items():
        match = re.search(pattern, report, re.IGNORECASE)
        if match:
            structured[key] = match.group(1).strip()
    
    # Sources with citation keys
    for i, src in enumerate(state.search_results, 1):
        structured["sources"].append({
            "citation_number": i,
            "title": src.get("title", "Unknown"),
            "url": src.get("url", ""),
            "source_type": src.get("source_type", "wiki"),
            "year": datetime.now().year,
        })
    
    return {
        "report": report,
        "structured_report": structured,
        "sources": state.search_results,
        "iteration": state.iteration + 1
    }

def should_continue(state: ResearchState) -> str:
    return "end" if state.iteration >= state.max_iterations else "continue"

workflow = StateGraph(ResearchState)
for name, fn in [
    ("planner", planner_node), 
    ("searcher", searcher_node), 
    ("indexer", indexer_node),
    ("analyzer", analyzer_node), 
    ("reporter", reporter_node)
]:
    workflow.add_node(name, fn)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "searcher")
workflow.add_edge("searcher", "indexer")
workflow.add_edge("indexer", "analyzer")
workflow.add_edge("analyzer", "reporter")
workflow.add_conditional_edges("reporter", should_continue, {"continue": "planner", "end": END})

research_graph = workflow.compile()

def run_deep_research(topic: str, max_iterations: int = 2) -> str:
    """Run deep research and return the final report."""
    result = research_graph.invoke(ResearchState(topic=topic, max_iterations=max_iterations))
    return result.get("report", "Research failed to produce a report.")

def run_deep_research_structured(topic: str, max_iterations: int = 2) -> Dict[str, Any]:
    """Run deep research and return structured output."""
    result = research_graph.invoke(ResearchState(topic=topic, max_iterations=max_iterations))
    return {
        "topic": topic,
        "report": result.get("report", ""),
        "structured_report": result.get("structured_report", {}),
        "sources": result.get("sources", []),
        "search_results": result.get("search_results", []),
        "timestamp": datetime.now().isoformat(),
    }

def run_research_with_paper(topic: str, max_iterations: int = 2, output_dir: str = "./paper_output") -> Dict[str, Any]:
    """Run deep research and generate a paper."""
    result = run_deep_research_structured(topic, max_iterations)
    
    # Generate paper
    try:
        from paper_gen import generate_research_paper
        paper = generate_research_paper(
            topic=topic,
            research_report=result["report"],
            sources=result["sources"],
            output_dir=output_dir
        )
        result["paper"] = {
            "title": paper.title,
            "output_dir": output_dir,
            "sections": len(paper.sections),
            "citations": len(paper.citations),
            "figures": len(paper.figures),
        }
    except Exception as e:
        console.print(f"[yellow]Paper generation failed:[/yellow] {e}")
        result["paper"] = {"error": str(e)}
    
    # Save structured output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # JSON
    (output_path / "research_output.json").write_text(json.dumps(result, indent=2))
    
    # Markdown
    md = f"# {topic}\n\n## Report\n{result['report']}\n\n## Sources\n"
    for i, src in enumerate(result["sources"], 1):
        md += f"{i}. {src.get('title', 'Unknown')} - {src.get('url', '')}\n"
    (output_path / "research_output.md").write_text(md)
    
    # BibTeX
    bib = ""
    for i, src in enumerate(result["sources"], 1):
        key = re.sub(r'[^\w]', '', src.get('title', f'source{i}')[:30]).lower()
        bib += f"@article{{{key}{src.get('year', datetime.now().year)},\n"
        bib += f"  title = {{{src.get('title', 'Unknown')}}},\n"
        bib += f"  author = {{Research Agent}},\n"
        bib += f"  journal = {{{src.get('source_type', 'wiki').capitalize()}}},\n"
        bib += f"  year = {{{src.get('year', datetime.now().year)}}},\n"
        bib += f"  url = {{{src.get('url', '')}}}\n"
        bib += "}\n\n"
    (output_path / "references.bib").write_text(bib)
    
    console.print(f"[green]Structured output saved to {output_path}[/green]")
    
    return result