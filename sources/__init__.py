from __future__ import annotations

import json
import re
import httpx
from dataclasses import dataclass
from typing import List, Optional
from bs4 import BeautifulSoup
from rich.console import Console

console = Console()

@dataclass
class SourceResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    source_type: str = "web"

class ArXivSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "http://export.arxiv.org/api/query"

    def search(self, query: str) -> List[SourceResult]:
        console.print(f"[cyan]Searching arXiv:[/cyan] {query}")
        try:
            import urllib.parse
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            resp = httpx.get(self.base_url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            
            # Parse XML
            soup = BeautifulSoup(resp.text, "xml")
            results = []
            for entry in soup.find_all("entry"):
                title = entry.title.get_text(strip=True) if entry.title else ""
                url = entry.id.get_text(strip=True) if entry.id else ""
                summary = entry.summary.get_text(strip=True) if entry.summary else ""
                
                # Get PDF link
                pdf_url = ""
                for link in entry.find_all("link"):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break
                
                if title and url:
                    results.append(SourceResult(
                        title=title,
                        url=pdf_url or url,
                        snippet=summary[:300],
                        content="",
                        source_type="arxiv"
                    ))
            return results
        except Exception as e:
            console.print(f"[red]arXiv search error:[/red] {e}")
            return []

    def fetch_content(self, url: str) -> str:
        # For arXiv, we already have the abstract; could download PDF for full text
        return ""

class SemanticScholarSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    def search(self, query: str) -> List[SourceResult]:
        console.print(f"[cyan]Searching Semantic Scholar:[/cyan] {query}")
        try:
            params = {
                "query": query,
                "limit": self.max_results,
                "fields": "title,url,abstract,venue,year,citationCount,openAccessPdf",
            }
            headers = {"User-Agent": "ResearchAgent/1.0"}
            resp = httpx.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for paper in data.get("data", []):
                title = paper.get("title", "")
                url = paper.get("url", "")
                abstract = paper.get("abstract", "")
                venue = paper.get("venue", "")
                year = paper.get("year", "")
                
                pdf_url = ""
                if paper.get("openAccessPdf"):
                    pdf_url = paper["openAccessPdf"].get("url", "")
                
                if title:
                    results.append(SourceResult(
                        title=f"{title} ({venue}, {year})" if venue else title,
                        url=pdf_url or url,
                        snippet=abstract[:300] if abstract else "",
                        content="",
                        source_type="semantic_scholar"
                    ))
            return results
        except Exception as e:
            console.print(f"[red]Semantic Scholar error:[/red] {e}")
            return []

class GitHubSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.github.com/search/repositories"

    def search(self, query: str) -> List[SourceResult]:
        console.print(f"[cyan]Searching GitHub:[/cyan] {query}")
        try:
            params = {
                "q": f"{query} in:name,description,readme",
                "sort": "stars",
                "order": "desc",
                "per_page": self.max_results,
            }
            headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ResearchAgent/1.0"}
            resp = httpx.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for repo in data.get("items", []):
                title = repo.get("full_name", "")
                url = repo.get("html_url", "")
                description = repo.get("description", "")
                stars = repo.get("stargazers_count", 0)
                
                if title:
                    results.append(SourceResult(
                        title=f"{title} ({stars} ⭐)",
                        url=url,
                        snippet=description[:300] if description else "",
                        content="",
                        source_type="github"
                    ))
            return results
        except Exception as e:
            console.print(f"[red]GitHub error:[/red] {e}")
            return []

class StackOverflowSearcher:
    def __init__(self, max_results: int = 5, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout
        self.base_url = "https://api.stackexchange.com/2.3/search/advanced"

    def search(self, query: str) -> List[SourceResult]:
        console.print(f"[cyan]Searching Stack Overflow:[/cyan] {query}")
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": "stackoverflow",
                "pagesize": self.max_results,
                "filter": "withbody",
            }
            headers = {"User-Agent": "ResearchAgent/1.0"}
            resp = httpx.get(self.base_url, params=params, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            
            results = []
            for item in data.get("items", []):
                title = item.get("title", "")
                url = item.get("link", "")
                body = item.get("body", "")
                score = item.get("score", 0)
                
                # Clean HTML from body
                soup = BeautifulSoup(body, "lxml")
                snippet = soup.get_text(strip=True)[:300]
                
                if title:
                    results.append(SourceResult(
                        title=f"{title} ({score} votes)",
                        url=url,
                        snippet=snippet,
                        content="",
                        source_type="stackoverflow"
                    ))
            return results
        except Exception as e:
            console.print(f"[red]Stack Overflow error:[/red] {e}")
            return []

# Factory
def search_all_sources(query: str, max_per_source: int = 3) -> List[SourceResult]:
    """Search all available sources and combine results."""
    all_results = []
    for searcher_class in [ArXivSearcher, SemanticScholarSearcher, GitHubSearcher, StackOverflowSearcher]:
        searcher = searcher_class(max_results=max_per_source)
        all_results.extend(searcher.search(query))
    return all_results