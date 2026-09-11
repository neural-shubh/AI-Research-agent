# 🔎 AI Research Agent

> **An autonomous, locally-running research agent that plans investigations, searches the web, builds a persistent knowledge base, retrieves relevant evidence, and generates structured research reports.**

Built with **LangGraph, LangChain, Ollama, web retrieval, embeddings, and semantic search**.

---

## Overview

AI Research Agent is a local research workflow designed to turn a high-level research topic into a structured, multi-step investigation.

Instead of simply sending a query to an LLM and generating an answer, the system:

1. **Plans** the research
2. Breaks the topic into specific questions
3. Generates targeted search queries
4. Searches and fetches relevant web pages
5. Chunks and embeds retrieved information
6. Stores the information in a persistent vector index
7. Retrieves relevant context for each research question
8. Generates a structured research report
9. Can iterate through the research workflow multiple times

The result is a lightweight **agentic research pipeline with persistent retrieval**.

---

## ✨ Features

### 🧠 Autonomous Research Planning

The agent converts a research topic into a structured plan containing:

* Specific research questions
* Search queries
* Research depth

This allows the research process to be driven by the topic rather than relying on a single manually-written prompt.

### 🌐 Web Research

The built-in web searcher:

* Searches the web
* Retrieves result pages
* Extracts readable page content
* Preserves source URLs
* Feeds retrieved information into the research pipeline

The current implementation uses DuckDuckGo HTML search with `httpx` and `BeautifulSoup`.

### 📚 Persistent Knowledge Base

Retrieved web content is transformed into searchable knowledge.

The pipeline:

```text
Web Pages
    ↓
Content Extraction
    ↓
Text Chunking
    ↓
Embeddings
    ↓
Vector Store
```

Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`.

The vector index and metadata are persisted under:

```text
data/embeddings/
```

### 🔍 Semantic Retrieval

Instead of relying only on keyword matching, the agent converts queries into embeddings and performs cosine-similarity retrieval against the stored knowledge.

This allows the system to retrieve context that is semantically relevant to a research question.

### 🔄 Iterative Research Workflow

The core research system is implemented as a **LangGraph state machine**:

```text
Planner
   ↓
Searcher
   ↓
Indexer
   ↓
Analyzer
   ↓
Reporter
   ↓
┌───────────────┐
│ Continue?     │
└───────┬───────┘
        │
        ├── Yes → Planner
        │
        └── No  → End
```

This enables multi-iteration research rather than a single search-and-answer operation.

### 🤖 Local LLM Inference

The reasoning and report-generation components use **Ollama**, allowing the language model layer to run locally.

The project currently uses `ChatOllama` through LangChain.

### 🖥️ CLI Interface

The project provides a command-line interface powered by **Typer** and **Rich**, including:

* Quick search
* Knowledge-base indexing
* Semantic question answering
* Deep research
* Knowledge-base statistics
* Knowledge-base clearing
* Interactive research sessions

---

## 🏗️ Architecture

```text
                         ┌──────────────────┐
                         │    User Topic    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Planner      │
                         │                  │
                         │ Questions       │
                         │ Search Queries  │
                         │ Research Depth  │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Searcher     │
                         │                  │
                         │ Web Search       │
                         │ Page Retrieval   │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Indexer      │
                         │                  │
                         │ Chunking         │
                         │ Embeddings       │
                         │ Persistent Store │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Analyzer     │
                         │                  │
                         │ Semantic Search  │
                         │ Context Retrieval│
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │     Reporter     │
                         │                  │
                         │ Structured Report│
                         │ Sources          │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Iteration Check  │
                         └────────┬─────────┘
                                  │
                         ┌────────┴────────┐
                         │                 │
                      Continue            End
                         │                 │
                         └──→ Planner       ▼
                                      Final Report
```

---

## 🧩 Tech Stack

| Component        | Technology              |
| ---------------- | ----------------------- |
| Language         | Python                  |
| Agent Workflow   | LangGraph               |
| LLM Framework    | LangChain               |
| Local LLM        | Ollama                  |
| Embeddings       | Sentence Transformers   |
| Vector Retrieval | NumPy cosine similarity |
| Web Search       | DuckDuckGo              |
| Web Scraping     | BeautifulSoup + HTTPX   |
| CLI              | Typer + Rich            |
| Data Validation  | Pydantic                |

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
.\venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and configure Ollama

Install Ollama and make sure the Ollama service is available locally.

Pull the model configured for your environment.

Example:

```bash
ollama pull llama3.1:8b
```

---

## 💻 Usage

### Deep Research

Run a multi-step research workflow:

```bash
python -m main research "Large Language Model agents"
```

Specify the number of research iterations:

```bash
python -m main research "Computer Vision Foundation Models" --iterations 3
```

---

### Quick Web Search

Perform a direct web search and inspect retrieved pages:

```bash
python -m main quick "Python async programming"
```

Specify the number of results:

```bash
python -m main quick "Python async programming" --results 10
```

---

### Build the Knowledge Base

Search for a topic and add the retrieved information to the persistent knowledge base:

```bash
python -m main index "Python async best practices"
```

---

### Query the Knowledge Base

Ask a question against the indexed knowledge:

```bash
python -m main ask "What is asyncio?"
```

---

### View Knowledge-Base Statistics

```bash
python -m main stats
```

---

### Clear the Knowledge Base

```bash
python -m main clear
```

---

### Interactive Mode

Launch the interactive research interface:

```bash
python -m main
```

Available commands:

```text
research <topic>
quick <query>
ask <question>
index <topic>
stats
clear
quit
```

---

## 🔬 Research Pipeline

For a deep-research request, the agent executes the following workflow:

### 1. Planning

The LLM creates a research plan containing:

```text
Topic
├── Research Questions
├── Search Queries
└── Research Depth
```

### 2. Search

Each generated query is executed against the web and relevant pages are fetched.

### 3. Indexing

Retrieved content is:

* Cleaned
* Split into overlapping chunks
* Embedded
* Stored with source metadata

### 4. Retrieval

Each research question is converted into an embedding and compared against the knowledge base.

The most relevant chunks are retrieved using cosine similarity.

### 5. Reporting

The retrieved context is passed to the local LLM to generate a structured report containing:

* Executive Summary
* Key Findings
* Detailed Analysis
* Conclusions
* Sources

### 6. Iteration

The workflow can repeat the research cycle for additional depth.

---

## 📁 Project Structure

```text
.
├── main.py
│   └── CLI interface and user commands
│
├── research_graph.py
│   └── LangGraph research workflow
│
├── tools.py
│   ├── Web search
│   ├── Page retrieval
│   ├── Text chunking
│   ├── Embedding generation
│   ├── Vector storage
│   └── Semantic retrieval
│
├── requirements.txt
│   └── Python dependencies
│
└── data/
    └── embeddings/
        ├── embeddings.npy
        └── metadata.json
```

---

## 🔑 Design Principles

### Local-first

The language-model layer is designed around local Ollama inference rather than requiring a hosted LLM API.

### Persistent Retrieval

Research doesn't have to disappear after a single interaction. Retrieved information can be persisted and queried later through the knowledge base.

### Modular Architecture

The system separates:

* CLI interaction
* Research orchestration
* Web retrieval
* Embedding generation
* Knowledge storage
* Context retrieval
* Report generation

This makes individual components easier to extend or replace.

### Agentic Workflow

The system is structured as a stateful workflow rather than a single LLM call, allowing research planning, tool execution, retrieval, analysis, reporting, and iteration to be composed into one pipeline.

---

## 🛠️ Possible Extensions

The architecture provides a foundation for further development, including:

* Source credibility scoring
* Parallel web searches
* Duplicate-document detection
* Citation verification
* PDF/document ingestion
* Academic-paper search
* Better vector databases
* Reranking models
* Human-in-the-loop research approval
* Research history and experiment tracking
* Web UI for interactive research
* Additional LLM providers

---

## ⚠️ Current Limitations

This project is intentionally lightweight and has several areas that can be improved.

* Web retrieval relies on HTML pages and may fail on JavaScript-heavy websites.
* Search quality depends on the external search results returned.
* Citation generation is LLM-driven rather than independently verified.
* The vector store is currently implemented with NumPy rather than a dedicated vector database.
* Research quality depends on the selected local language model.
* The current workflow can revisit the same research cycle during iterations.

---

## 🎯 Why This Project?

Most LLM applications stop at:

```text
Question → LLM → Answer
```

This project explores a different approach:

```text
Question
   ↓
Research Planning
   ↓
Tool Use
   ↓
Web Retrieval
   ↓
Knowledge Construction
   ↓
Semantic Retrieval
   ↓
Analysis
   ↓
Report Generation
   ↓
Iterative Research
```

The goal is to explore how **agentic workflows, retrieval systems, and local language models** can be combined to create a more autonomous research system.

---

## 📌 Status

**Active project / research prototype**

The core research pipeline is functional, while retrieval quality, source validation, and agent autonomy remain areas for future development.

---

