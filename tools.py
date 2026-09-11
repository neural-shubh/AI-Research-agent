from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from bs4 import BeautifulSoup
from fastembed import TextEmbedding
from langchain_core.tools import tool
from rich.console import Console
from tqdm import tqdm

console = Console()


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content[:5000],
        }


class WebSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 10):
        self.max_results = max_results
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        }

    def search(self, query: str) -> List[SearchResult]:
        console.print(f"[cyan]Searching:[/cyan] {query}")
        try:
            import httpx
            encoded = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded}"
            resp = httpx.get(url, headers=self.headers, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one("a.result__snippet")
                url_elem = result.select_one(".result__url")
                if title_elem and title_elem.get("href"):
                    title = title_elem.get_text(strip=True)
                    link = title_elem["href"]
                    if link.startswith("//duckduckgo.com/l/"):
                        parsed = urllib.parse.urlparse(link)
                        params = urllib.parse.parse_qs(parsed.query)
                        if "uddg" in params:
                            link = params["uddg"][0]
                    snippet = ""
                    if url_elem:
                        snippet = url_elem.get_text(strip=True)
                    if title and link:
                        results.append(SearchResult(title=title, url=link, snippet=snippet))
                        if len(results) >= self.max_results:
                            break
            return results
        except Exception as e:
            console.print(f"[red]Search error:[/red] {e}")
            return []

    def fetch_content(self, url: str) -> str:
        try:
            import httpx
            resp = httpx.get(url, headers=self.headers, timeout=self.timeout, follow_redirects=True)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            return text[:8000]
        except Exception as e:
            console.print(f"[yellow]Fetch failed for {url}:[/yellow] {e}")
            return ""

    def search_and_fetch(self, query: str) -> List[SearchResult]:
        results = self.search(query)
        for r in tqdm(results, desc="Fetching pages", leave=False):
            r.content = self.fetch_content(r.url)
        return results


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
        for doc in documents:
            if not doc.content.strip():
                continue
            chunks = self._chunk_text(doc.content, doc.url)
            for c in chunks:
                c["title"] = doc.title
                c["url"] = doc.url
            all_chunks.extend(chunks)

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
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if idx < len(self._metadata):
                meta = self._metadata[idx].copy()
                meta["score"] = float(scores[idx])
                results.append(meta)
        return results

    def save(self):
        if self._embeddings:
            np.save(self.persist_dir / "embeddings.npy", np.array(self._embeddings))
        import json
        with open(self.persist_dir / "metadata.json", "w") as f:
            json.dump(self._metadata, f)
        console.print(f"[green]Saved index:[/green] {len(self._embeddings)} vectors")

    def load(self) -> bool:
        import json
        emb_path = self.persist_dir / "embeddings.npy"
        meta_path = self.persist_dir / "metadata.json"
        if emb_path.exists() and meta_path.exists():
            self._embeddings = list(np.load(emb_path))
            with open(meta_path) as f:
                self._metadata = json.load(f)
            console.print(f"[green]Loaded index:[/green] {len(self._embeddings)} vectors")
            return True
        return False

    @property
    def index_count(self) -> int:
        return len(self._embeddings)


web_searcher = WebSearcher()
embedding_store = EmbeddingStore()


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
        out.append(f"[{i}] {r['title']} (score: {r['score']:.3f})\n{r['text'][:800]}...\nSource: {r['url']}")
    return "\n\n".join(out)


@tool
def add_to_knowledge_base(query: str) -> str:
    """Search for a topic and add results to the knowledge base."""
    results = web_searcher.search_and_fetch(query)
    count = embedding_store.add_documents(results)
    embedding_store.save()
    return f"Added {count} chunks from {len(results)} pages to knowledge base."