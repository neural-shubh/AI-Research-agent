# 🔬 AI Research Agent

> **Turn a research topic into a structured research report — locally.**

An autonomous research agent that **plans investigations, searches multiple sources, ingests documents, builds a persistent knowledge base, retrieves relevant evidence, and generates structured research reports and papers.**

Built with **LangGraph + LangChain + Ollama**, with a local-first architecture and no external LLM API required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-orange)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)

---

## 🧠 Why I Built This

Writing a research paper isn't just writing the paper.

You have to:

* 🔎 Find relevant papers and sources
* 📚 Read through large amounts of information
* 🧩 Break a topic into smaller research questions
* 🗂️ Keep track of useful information
* 🔗 Connect evidence across different sources
* ✍️ Turn everything into a structured report

So instead of repeatedly doing those steps manually, I built an agent that can **orchestrate the research workflow for me.**

The goal isn't simply *"ask an LLM a question."*

It's:

> **Give the agent a topic → let it investigate → build knowledge → retrieve evidence → produce a research report.**

---

# ✨ What It Can Do

| Capability                | What it does                                                  |
| ------------------------- | ------------------------------------------------------------- |
| 🔎 **Quick Search**       | Search multiple sources and generate cited answers            |
| 🧠 **Semantic Q&A**       | Ask questions against your accumulated knowledge base         |
| 📚 **Knowledge Base**     | Persistently store and retrieve research information          |
| 📄 **Document Ingestion** | Ingest PDF, Markdown, Python, Jupyter Notebook and text files |
| 🤖 **Deep Research**      | Perform multi-step research using a LangGraph workflow        |
| 📝 **Paper Generation**   | Generate LaTeX, Markdown and BibTeX research outputs          |
| 🖼️ **Figure Extraction** | Extract figures from research PDFs                            |
| 🌐 **Web UI**             | Use the entire system through a browser interface             |
| 🔒 **Local-First**        | Run the LLM locally through Ollama                            |

---

# 🚀 The Research Workflow

```text
                         ┌──────────────┐
                         │  Research    │
                         │    Topic     │
                         └──────┬───────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Planner     │
                       │ Questions +     │
                       │ Search Queries  │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    Searcher     │
                       │ Multiple Sources│
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │     Indexer     │
                       │ Chunk + Embed   │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    Analyzer     │
                       │ Retrieve +      │
                       │ Analyze Evidence│
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │    Reporter     │
                       │ Structured      │
                       │ Research Report │
                       └────────┬────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Iterate /      │
                       │  Refine Research│
                       └─────────────────┘
```

The workflow is orchestrated using **LangGraph**, allowing the research process to move through multiple stages instead of relying on a single LLM prompt.

---

# 🌐 Research Sources

The agent can work with multiple information sources, including:

* Wikipedia
* arXiv
* GitHub
* Semantic Scholar
* Stack Overflow

This allows research to combine information from different types of sources rather than relying on a single search result.

---

# 📚 Persistent Knowledge Base

Research doesn't have to disappear after one question.

The agent maintains a persistent local knowledge base where documents can be:

```text
Document
   │
   ▼
Chunk
   │
   ▼
Embedding
   │
   ▼
Vector Store
   │
   ▼
Semantic Retrieval
```

You can then ask questions against previously indexed information.

For example:

```bash
.\run.bat index "Vision Transformers"

.\run.bat ask "How do Vision Transformers differ from CNNs?"
```

The system retrieves the most relevant chunks based on semantic similarity.

---

# 📄 Document Ingestion

You can also bring your own research material into the system.

Supported formats include:

```text
PDF
Markdown
Python
Jupyter Notebook
TXT
```

Directories can be ingested recursively:

```bash
.\run.bat ingest ./research_material
```

This makes it possible to build a personal research library and query it later.

---

# 📝 Research Paper Generation

The research agent can turn its findings into structured paper outputs.

```bash
.\run.bat research "Vision Transformers" -i 1 --paper
```

Generated output:

```text
paper_output_<job_id>/
│
├── main.tex
├── main.md
├── references.bib
├── paper_metadata.json
│
└── figures/
    └── extracted_figures...
```

### Output includes

* 📑 arXiv-style LaTeX
* 📝 Markdown version
* 📚 BibTeX references
* 🖼️ Extracted figures
* 🗃️ Paper metadata

---

# ⚡ Quick Start

### 1. Clone the repository

```bash
git clone <your-repo>
cd ai-research-agent
```

### 2. Create the environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
.\venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Ollama

Install Ollama and pull the model you want to use.

```bash
ollama pull phi3:3.8b
```

### 5. Start researching

```bash
.\run.bat quick "vision transformers"

.\run.bat ask "what is multi-head attention?"

.\run.bat research "ViT vs CNN" -i 2
```

---

# 🖥️ Web Interface

Prefer a browser instead of the CLI?

Start the Web UI:

```bash
python -m uvicorn web_ui:app --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

### Available sections

**🔎 Quick Search**
Search the web and generate an answer.

**🧠 Ask KB**
Query your indexed research knowledge.

**🤖 Deep Research**
Run multi-step research with progress tracking.

**📥 Ingest**
Add local research documents.

**⚙️ Manage**
View knowledge-base statistics, clear data and index new topics.

---

# 💻 CLI Reference

| Command                         | Purpose                                      |
| ------------------------------- | -------------------------------------------- |
| `quick "query"`                 | Search + generate a cited answer             |
| `ask "question"`                | Query the knowledge base                     |
| `index "topic"`                 | Search and add a topic to the knowledge base |
| `ingest ./folder`               | Recursively ingest local documents           |
| `research "topic" -i 2`         | Run deep research                            |
| `research "topic" -i 1 --paper` | Generate a research paper                    |
| `stats`                         | Display knowledge-base statistics            |
| `interactive`                   | Start interactive REPL mode                  |

---

# 🏗️ Architecture

```text
                         AI RESEARCH AGENT
                                │
              ┌─────────────────┴─────────────────┐
              │                                   │
              ▼                                   ▼
        Quick / Ask                         Deep Research
              │                                   │
              ▼                                   ▼
        Web Searchers                         Planner
              │                                   │
              ▼                                   ▼
        Fetch + Chunk                         Searcher
              │                                   │
              ▼                                   ▼
           Embed                              Indexer
              │                                   │
              ▼                                   ▼
       Vector Store                           Analyzer
              │                                   │
              │                                   ▼
              │                                Reporter
              │                                   │
              │                                   ▼
              │                           Paper Generator
              │                                   │
              └───────────────┬───────────────────┘
                              ▼
                    Persistent Knowledge Base
```

### Core components

| Component           | Technology              |
| ------------------- | ----------------------- |
| Agent orchestration | LangGraph               |
| LLM integration     | LangChain + Ollama      |
| Embeddings          | fastembed               |
| Similarity search   | NumPy cosine similarity |
| Web retrieval       | HTTPX + BeautifulSoup   |
| Document processing | PyMuPDF                 |
| API / Web backend   | FastAPI                 |
| Frontend            | HTMX                    |
| CLI                 | Typer + Rich            |
| Data validation     | Pydantic                |

---

# ⚙️ Configuration

The system can be configured through environment variables.

| Variable       | Default                                  | Purpose         |
| -------------- | ---------------------------------------- | --------------- |
| `OLLAMA_HOST`  | `http://localhost:11434`                 | Ollama server   |
| `OLLAMA_MODEL` | `phi3:3.8b`                              | LLM model       |
| `EMBED_MODEL`  | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |

Example:

```env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=phi3:3.8b
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

# 📁 Project Structure

```text
ai-research-agent/
│
├── main.py
├── research_graph.py
├── tools.py
├── web_ui.py
├── requirements.txt
├── run.bat
├── .env
│
├── templates/
│   └── ...
│
├── data/
│   └── embeddings/
│
└── paper_output_<job_id>/
    ├── main.tex
    ├── main.md
    ├── references.bib
    ├── paper_metadata.json
    └── figures/
```

---

# 🔍 Example

### Input

```text
"How do Vision Transformers compare with CNNs?"
```

### The agent

```text
1. Creates research questions
2. Generates search queries
3. Searches multiple sources
4. Retrieves relevant information
5. Chunks and embeds the results
6. Stores the information
7. Retrieves relevant evidence
8. Analyzes the findings
9. Generates a structured report
10. Optionally generates a paper
```

### Example report structure

```text
Executive Summary

Key Findings
    [1] ...
    [2] ...
    [3] ...

Detailed Analysis

Conclusions

References
    [1] ...
    [2] ...
```

---

# 🧩 Design Principles

### Local-first

The system is designed around local LLM inference through Ollama rather than requiring an external LLM API.

### Research-oriented

The goal is not just question answering. The workflow is designed around **planning, retrieval, evidence gathering and synthesis**.

### Persistent

Research can be stored and queried again instead of starting from zero every time.

### Modular

Search, ingestion, embeddings, retrieval, analysis and reporting are separated into different components, making the system easier to extend.

### Agentic

The system uses a multi-step graph rather than treating research as a single prompt-response interaction.

---

# 🚧 Current Limitations

This project is still evolving.

Some areas that can be improved include:

* Better source ranking and validation
* More robust citation verification
* Improved long-document retrieval
* More sophisticated research planning
* Better handling of conflicting sources
* Additional academic databases
* More advanced paper formatting
* Automated experiment / literature tracking

---

# 🛣️ Possible Future Improvements

```text
                    Current
                       │
                       ▼
              ┌────────────────┐
              │ Research Agent │
              └───────┬────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
      Better RAG   Better Web   Better
      Retrieval    Research    Citations
          │           │           │
          └───────────┼───────────┘
                      ▼
              ┌────────────────┐
              │ Research       │
              │ Assistant      │
              └────────────────┘
```

Potential directions include:

* 🔬 Research-paper discovery
* 🧠 Improved RAG pipelines
* 📊 Research comparison tools
* 🔗 Citation graphs
* 🧪 Experiment tracking
* 📑 Automatic literature reviews
* 🤝 Multi-agent research workflows

---

# 📌 Project Status

**Active development 🚧**

The core research workflow is functional, with web search, document ingestion, persistent retrieval, deep research and paper-generation capabilities.

The architecture is intentionally modular so additional research tools and workflows can be added over time.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests where appropriate
5. Open a pull request

Ideas for new search sources, retrieval improvements and research workflows are especially welcome.

---

### 🔬 Built to make research less painful.

**Python · LangGraph · LangChain · Ollama · FastAPI · HTMX · fastembed · PyMuPDF**
