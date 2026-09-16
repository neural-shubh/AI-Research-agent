from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Set

import numpy as np
from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from rich.console import Console
from tqdm import tqdm

console = Console()

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    source_type: str = "wiki"
    def to_dict(self) -> dict:
        return {"title": self.title, "url": self.url, "snippet": self.snippet, "content": self.content[:5000], "source_type": self.source_type}

class WebSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ResearchAgent/1.0 (https://github.com/username/research-agent)",
            "Accept": "application/json",
        }
        self._seen_urls: Set[str] = set()

    def _expand_query(self, query: str) -> List[str]:
        query_lower = query.lower()
        expansions = [query]
        
        if "vision transformer" in query_lower or "vit" in query_lower:
            expansions.extend([
                "Vision Transformer (ViT)",
                "Vision Transformer deep learning",
                "An Image is Worth 16x16 Words",
                "ViT architecture",
            ])
        elif "transformer" in query_lower and "vision" not in query_lower:
            expansions.append("Transformer (deep learning)")
        elif "llm" in query_lower or "large language model" in query_lower:
            expansions.extend(["Large language model", "LLM architecture", "Transformer (deep learning)"])
        elif "attention" in query_lower:
            expansions.extend(["Attention mechanism", "Multi-head attention", "Self-attention"])
        elif "diffusion" in query_lower:
            expansions.extend(["Diffusion model", "Stable Diffusion", "Denoising diffusion"])
        elif "rag" in query_lower or "retrieval augmented" in query_lower:
            expansions.extend(["Retrieval-Augmented Generation", "RAG architecture", "Vector database"])
            
        return expansions

    def search(self, query: str) -> List[SearchResult]:
        console.print(f"[cyan]Searching Wikipedia:[/cyan] {query}")
        all_results = []
        self._seen_urls.clear()
        
        for expanded_query in self._expand_query(query):
            try:
                import httpx
                url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": expanded_query,
                    "format": "json",
                    "srlimit": self.max_results,
                }
                resp = httpx.get(url, params=params, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("query", {}).get("search", []):
                    title = item["title"]
                    wiki_url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
                    if wiki_url in self._seen_urls:
                        continue
                    self._seen_urls.add(wiki_url)
                    snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                    all_results.append(SearchResult(title=title, url=wiki_url, snippet=snippet, source_type="wiki"))
                    if len(all_results) >= self.max_results:
                        break
                if len(all_results) >= self.max_results:
                    break
            except Exception as e:
                console.print(f"[red]Search error for '{expanded_query}':[/red] {e}")
        return all_results

    def fetch_content(self, url: str) -> str:
        try:
            import httpx
            if "en.wikipedia.org/wiki/" in url:
                title = url.split("/wiki/")[-1]
                api_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json",
                }
                resp = httpx.get(api_url, params=params, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                pages = data.get("query", {}).get("pages", {})
                for page in pages.values():
                    if "extract" in page:
                        return page["extract"][:8000]
            
            resp = httpx.get(url, headers=self.headers, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)[:8000]
        except Exception as e:
            console.print(f"[yellow]Fetch failed for {url}:[/yellow] {e}")
            return ""

    def search_and_fetch(self, query: str) -> List[SearchResult]:
        results = self.search(query)
        for r in tqdm(results, desc="Fetching pages", leave=False):
            r.content = self.fetch_content(r.url)
        return results


class LocalDocumentIngester:
    """Ingest local documents (PDF, MD, .py, .ipynb) into the knowledge base."""
    
    SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".py", ".ipynb", ".txt", ".rst"}
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _extract_pdf(self, path: Path) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            console.print(f"[red]PDF extraction failed for {path}:[/red] {e}")
            return ""
    
    def _extract_ipynb(self, path: Path) -> str:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            text = ""
            for cell in data.get("cells", []):
                if cell.get("cell_type") in ["markdown", "code"]:
                    text += "".join(cell.get("source", [])) + "\n\n"
            return text
        except Exception as e:
            console.print(f"[red]Notebook extraction failed for {path}:[/red] {e}")
            return ""
    
    def _extract_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            console.print(f"[red]Text extraction failed for {path}:[/red] {e}")
            return ""
    
    def extract(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf(path)
        elif suffix == ".ipynb":
            return self._extract_ipynb(path)
        else:
            return self._extract_text(path)
    
    def _chunk_text(self, text: str, source: str) -> List[dict]:
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": len(chunks),
            })
        return chunks
    
    def ingest_directory(self, directory: Path, recursive: bool = True, embedding_store=None) -> int:
        """Ingest all supported documents from a directory."""
        if embedding_store is None:
            from tools import embedding_store as es
            embedding_store = es
        
        pattern = "**/*" if recursive else "*"
        files = []
        for ext in self.SUPPORTED_EXTENSIONS:
            files.extend(directory.glob(pattern + ext))
        
        if not files:
            console.print(f"[yellow]No supported files found in {directory}[/yellow]")
            return 0
        
        console.print(f"[cyan]Found {len(files)} files to ingest[/cyan]")
        total_chunks = 0
        
        for file_path in tqdm(files, desc="Ingesting files"):
            try:
                content = self.extract(file_path)
                if not content.strip():
                    continue
                
                rel_path = file_path.relative_to(directory)
                chunks = self._chunk_text(content, str(rel_path))
                for c in chunks:
                    c["title"] = file_path.name
                    c["url"] = f"file://{file_path}"
                    c["source_type"] = "local"
                
                texts = [c["text"] for c in chunks]
                embeddings = list(embedding_store.model.embed(texts, batch_size=32))
                embeddings = np.array(embeddings, dtype=np.float32)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norms + 1e-10)
                
                embedding_store._embeddings.extend(embeddings)
                embedding_store._metadata.extend(chunks)
                total_chunks += len(chunks)
                embedding_store._indexed_urls.add(f"file://{file_path}")
                
            except Exception as e:
                console.print(f"[red]Failed to ingest {file_path}:[/red] {e}")
        
        if total_chunks > 0:
            embedding_store.save()
            console.print(f"[green]Ingested {total_chunks} chunks from {len(files)} files[/green]")
        
        return total_chunks


class EmbeddingStore:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_dir: str = "./data/embeddings",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ):
        self.model_name = model_name
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model: Optional[TextEmbedding] = None
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[dict] = []
        self._indexed_urls: Set[str] = set()

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            console.print(f"[cyan]Loading embedding model:[/cyan] {self.model_name}")
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def _chunk_text(self, text: str, source: str) -> List[dict]:
        chunks = []
        words = text.split()
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            if len(chunk_words) < 50:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "text": chunk_text,
                "source": source,
                "chunk_id": len(chunks),
            })
        return chunks

    def add_documents(self, documents: List[SearchResult]) -> int:
        all_chunks = []
        new_urls = 0
        for doc in documents:
            if not doc.content.strip():
                continue
            if doc.url in self._indexed_urls:
                continue
            chunks = self._chunk_text(doc.content, doc.url)
            for c in chunks:
                c["title"] = doc.title
                c["url"] = doc.url
                c["source_type"] = doc.source_type
            all_chunks.extend(chunks)
            self._indexed_urls.add(doc.url)
            new_urls += 1

        if not all_chunks:
            return 0

        texts = [c["text"] for c in all_chunks]
        embeddings = list(self.model.embed(texts, batch_size=32))
        embeddings = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-10)

        self._embeddings.extend(embeddings)
        self._metadata.extend(all_chunks)
        return len(all_chunks)

    def search(self, query: str, k: int = 5) -> List[dict]:
        if not self._embeddings:
            return []
        q_emb = np.array(list(self.model.embed([query])))[0]
        q_emb = q_emb / (np.linalg.norm(q_emb) + 1e-10)
        scores = np.dot(np.array(self._embeddings), q_emb)
        top_indices = np.argsort(scores)[::-1][:k * 2]
        
        seen_urls = set()
        results = []
        for idx in top_indices:
            if idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                url = meta.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                meta["score"] = float(scores[idx])
                results.append(meta)
                if len(results) >= k:
                    break
        return results

    def save(self):
        if self._embeddings:
            np.save(self.persist_dir / "embeddings.npy", np.array(self._embeddings))
        with open(self.persist_dir / "metadata.json", "w") as f:
            json.dump(self._metadata, f)
        console.print(f"[green]Saved index:[/green] {len(self._embeddings)} vectors from {len(self._indexed_urls)} sources")

    def load(self) -> bool:
        emb_path = self.persist_dir / "embeddings.npy"
        meta_path = self.persist_dir / "metadata.json"
        if emb_path.exists() and meta_path.exists():
            self._embeddings = list(np.load(emb_path))
            with open(meta_path) as f:
                self._metadata = json.load(f)
            self._indexed_urls = set(m.get("url", "") for m in self._metadata)
            console.print(f"[green]Loaded index:[/green] {len(self._embeddings)} vectors from {len(self._indexed_urls)} sources")
            return True
        return False

    @property
    def index_count(self) -> int:
        return len(self._embeddings)


web_searcher = WebSearcher()
embedding_store = EmbeddingStore()
local_ingester = LocalDocumentIngester()


@tool
def web_search(query: str) -> str:
    """Search the web and return top results with fetched content."""
    results = web_searcher.search_and_fetch(query)
    return json.dumps([r.to_dict() for r in results], indent=2)


@tool
def retrieve_context(query: str, k: int = 5) -> str:
    """Retrieve relevant context from the vector store."""
    results = embedding_store.search(query, k=k)
    if not results:
        return "No relevant context found."
    out = []
    for i, r in enumerate(results, 1):
        source_icon = {"wiki": "[W]", "arxiv": "[A]", "semantic_scholar": "[S]", "github": "[G]", "stackoverflow": "[SO]", "local": "[L]"}.get(r.get('source_type', ''), '[D]')
        out.append(f"[{i}] {source_icon} {r['title']} (score: {r['score']:.3f})\n{r['text'][:800]}...\nSource: {r['url']}")
    return "\n\n".join(out)


@tool
def add_to_knowledge_base(query: str) -> str:
    """Search for a topic and add results to the knowledge base."""
    results = web_searcher.search_and_fetch(query)
    count = embedding_store.add_documents(results)
    embedding_store.save()
    return f"Added {count} chunks from {len(results)} new pages to knowledge base."


@tool
def ingest_local_documents(path: str, recursive: bool = True) -> str:
    """Ingest local documents (PDF, MD, .py, .ipynb) from a directory."""
    p = Path(path)
    if not p.exists():
        return f"Path not found: {path}"
    if not p.is_dir():
        return f"Path must be a directory: {path}"
    count = local_ingester.ingest_directory(p, recursive=recursive)
    return f"Ingested {count} chunks from local documents in {path}"


# Quick search with LLM synthesis
llm = ChatOllama(model="phi3:3.8b", temperature=0.3, num_predict=512)

def quick_search_with_answer(query: str, k: int = 5) -> dict:
    """Search and synthesize answer with citations."""
    results = web_searcher.search_and_fetch(query)
    if not results:
        return {"answer": "No results found.", "sources": []}
    
    # Prepare context
    context_parts = []
    for i, r in enumerate(results[:k], 1):
        context_parts.append(f"[{i}] {r.title}\n{r.content[:2000]}\nURL: {r.url}")
    context = "\n\n---\n\n".join(context_parts)
    
    prompt = f"""Answer the question based on the provided sources. Use numeric citations like [1], [2].
Question: {query}

Sources:
{context}

Answer:"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    sources = [{"title": r.title, "url": r.url, "source_type": r.source_type} for r in results[:k]]
    
    return {"answer": answer, "sources": sources}