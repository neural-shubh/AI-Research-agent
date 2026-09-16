from __future__ import annotations

import json
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

from research_graph import run_deep_research_structured, run_research_with_paper
from tools import quick_search_with_answer, retrieve_context, add_to_knowledge_base, ingest_local_documents, embedding_store, web_searcher

app = FastAPI(title="Local AI Research Agent")

# Templates - use custom environment to avoid caching issues
from jinja2 import Environment, FileSystemLoader
jinja_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=True,
    cache_size=0,  # Disable cache
)
# Disable cache completely
jinja_env.cache = {}

# Custom template response
async def template_response(template_name: str, context: dict, request: Request = None):
    """Render template without caching issues."""
    template = jinja_env.get_template(template_name)
    if request:
        context = {**context, "request": request}
    content = template.render(**context)
    return HTMLResponse(content=content)

# Keep Jinja2Templates for compatibility but don't use it
templates = None

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ResearchRequest(BaseModel):
    topic: str
    iterations: int = 2
    generate_paper: bool = False

class QuickSearchRequest(BaseModel):
    query: str
    k: int = 5
    answer: bool = True

class AskRequest(BaseModel):
    question: str
    k: int = 5

class IndexRequest(BaseModel):
    query: str

class IngestRequest(BaseModel):
    path: str
    recursive: bool = True

# In-memory job tracking
jobs = {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return await template_response("index.html", {"current_user": None}, request)

@app.post("/api/quick")
async def api_quick(request: QuickSearchRequest):
    result = quick_search_with_answer(request.query, k=request.k)
    if not request.answer:
        # Just return sources
        sources = web_searcher.search_and_fetch(request.query)
        return {"sources": [s.to_dict() for s in sources[:request.k]]}
    return result

@app.post("/api/ask")
async def api_ask(request: AskRequest):
    embedding_store.load()
    context = retrieve_context.invoke({"query": request.question, "k": request.k})
    return {"context": context}

@app.post("/api/index")
async def api_index(request: IndexRequest):
    result = add_to_knowledge_base.invoke({"query": request.query})
    return {"message": result}

@app.post("/api/ingest")
async def api_ingest(request: IngestRequest):
    embedding_store.load()
    result = ingest_local_documents(request.path, recursive=request.recursive)
    return {"message": result}

@app.post("/api/research")
async def api_research(request: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = f"research_{len(jobs)}"
    jobs[job_id] = {"status": "started", "topic": request.topic, "result": None}
    
    async def run_research():
        try:
            if request.generate_paper:
                result = run_research_with_paper(
                    request.topic, 
                    max_iterations=request.iterations,
                    output_dir=f"./paper_output_{job_id}"
                )
            else:
                result = run_deep_research_structured(request.topic, max_iterations=request.iterations)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    
    background_tasks.add_task(run_research)
    return {"job_id": job_id, "status": "started"}

@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}, 404
    return jobs[job_id]

@app.get("/api/stats")
async def api_stats():
    embedding_store.load()
    return {
        "vectors": embedding_store.index_count,
        "sources": len(embedding_store._metadata),
    }

@app.get("/api/clear")
async def api_clear():
    import shutil
    if embedding_store.persist_dir.exists():
        shutil.rmtree(embedding_store.persist_dir)
    embedding_store._embeddings = []
    embedding_store._metadata = []
    return {"message": "Knowledge base cleared"}

# HTMX endpoints for partial updates
@app.post("/search", response_class=HTMLResponse)
async def search_htmx(request: Request, query: str = Form(...), answer: bool = Form(True)):
    result = quick_search_with_answer(query)
    return await template_response("partials/search_results.html", {
        "result": result,
        "query": query
    }, request)

@app.post("/ask", response_class=HTMLResponse)
async def ask_htmx(request: Request, question: str = Form(...), k: int = Form(5)):
    embedding_store.load()
    context = retrieve_context.invoke({"query": question, "k": k})
    return await template_response("partials/ask_result.html", {
        "context": context,
        "question": question
    }, request)

@app.post("/research", response_class=HTMLResponse)
async def research_htmx(request: Request, topic: str = Form(...), iterations: int = Form(2), paper: bool = Form(False)):
    job_id = f"research_{len(jobs)}"
    jobs[job_id] = {"status": "started", "topic": topic, "result": None}
    
    async def run_research():
        try:
            if paper:
                result = run_research_with_paper(topic, max_iterations=iterations, output_dir=f"./paper_output_{job_id}")
            else:
                result = run_deep_research_structured(topic, max_iterations=iterations)
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
        except Exception as e:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
    
    asyncio.create_task(run_research())
    
    return await template_response("partials/job_status.html", {
        "job_id": job_id,
        "topic": topic
    }, request)

@app.get("/job/{job_id}/status", response_class=HTMLResponse)
async def job_status_htmx(request: Request, job_id: str):
    job = jobs.get(job_id, {"status": "unknown"})
    return await template_response("partials/job_status.html", {
        "job_id": job_id,
        "job": job
    }, request)

@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """Download generated paper files."""
    file_path = Path(f"./paper_output_{job_id}") / filename
    if file_path.exists():
        return FileResponse(file_path)
    return {"error": "File not found"}, 404

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)