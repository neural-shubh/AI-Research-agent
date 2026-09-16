from __future__ import annotations

import json
import re
import os
import tempfile
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import httpx
from bs4 import BeautifulSoup

# Try to import fitz for PDF figure extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

@dataclass
class Citation:
    key: str
    title: str
    authors: List[str]
    venue: str
    year: int
    url: str
    doi: Optional[str] = None

@dataclass
class Figure:
    caption: str
    image_path: str
    source_paper: str
    page_num: int

@dataclass
class PaperSection:
    name: str
    content: str
    citations: List[int] = field(default_factory=list)  # Indices into citations list
    figures: List[int] = field(default_factory=list)    # Indices into figures list

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    sections: List[PaperSection]
    citations: List[Citation]
    figures: List[Figure]
    keywords: List[str]
    
    def to_latex(self) -> str:
        """Generate arXiv-style LaTeX paper."""
        lines = [
            r"\documentclass[twocolumn,10pt]{article}",
            r"\usepackage{arxiv}",
            r"\usepackage{graphicx}",
            r"\usepackage{hyperref}",
            r"\usepackage{booktabs}",
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{natbib}",
            r"\usepackage{doi}",
            "",
            f"\\title{{{self._escape_latex(self.title)}}}",
            f"\\author{{{', '.join(self._escape_latex(a) for a in self.authors)}}}",
            f"\\date{{{datetime.now().strftime('%B %d, %Y')}}}",
            "",
            r"\begin{document}",
            r"\maketitle",
            "",
            r"\begin{abstract}",
            self._escape_latex(self.abstract),
            r"\end{abstract}",
            "",
            f"\\keywords{{{', '.join(self._escape_latex(k) for k in self.keywords)}}}",
            "",
        ]
        
        # Sections
        for section in self.sections:
            lines.append(f"\\section{{{self._escape_latex(section.name)}}}")
            lines.append("")
            content = self._escape_latex(section.content)
            # Replace citation markers [1] -> \cite{key}
            for i, cite_idx in enumerate(section.citations):
                if cite_idx < len(self.citations):
                    key = self.citations[cite_idx].key
                    content = content.replace(f"[{i+1}]", f"\\cite{{{key}}}")
            lines.append(content)
            lines.append("")
            
            # Figures
            for fig_idx in section.figures:
                if fig_idx < len(self.figures):
                    fig = self.figures[fig_idx]
                    lines.append(f"\\begin{{figure}}[htbp]")
                    lines.append(f"  \\centering")
                    lines.append(f"  \\includegraphics[width=0.9\\linewidth]{{{fig.image_path}}}")
                    lines.append(f"  \\caption{{{self._escape_latex(fig.caption)}}}")
                    lines.append(f"\\end{{figure}}")
                    lines.append("")
        
        # Bibliography
        lines.append(r"\bibliographystyle{unsrt}")
        lines.append(r"\bibliography{references}")
        lines.append("")
        lines.append(r"\end{document}")
        
        return "\n".join(lines)
    
    def to_markdown(self) -> str:
        """Generate Markdown version."""
        lines = [
            f"# {self.title}",
            "",
            f"**Authors:** {', '.join(self.authors)}",
            f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
            "",
            "## Abstract",
            self.abstract,
            "",
            f"**Keywords:** {', '.join(self.keywords)}",
            "",
        ]
        
        for section in self.sections:
            lines.append(f"## {section.name}")
            lines.append("")
            content = section.content
            # Replace citation markers
            for i, cite_idx in enumerate(section.citations):
                if cite_idx < len(self.citations):
                    key = self.citations[cite_idx].key
                    content = content.replace(f"[{i+1}]", f"[{cite_idx+1}]")
            lines.append(content)
            lines.append("")
            
            for fig_idx in section.figures:
                if fig_idx < len(self.figures):
                    fig = self.figures[fig_idx]
                    lines.append(f"![{fig.caption}]({fig.image_path})")
                    lines.append(f"*{fig.caption}*")
                    lines.append("")
        
        # References
        lines.append("## References")
        lines.append("")
        for i, cite in enumerate(self.citations, 1):
            lines.append(f"[{i}] {cite.authors[0] if cite.authors else 'Unknown'} et al. "
                         f"\"{cite.title}\". *{cite.venue}*, {cite.year}. "
                         f"DOI: {cite.doi or 'N/A'}. URL: {cite.url}")
        
        return "\n".join(lines)
    
    def to_bibtex(self) -> str:
        """Generate BibTeX file."""
        entries = []
        for cite in self.citations:
            entry = f"@article{{{cite.key},\n"
            entry += f"  title = {{{cite.title}}},\n"
            entry += f"  author = {{{' and '.join(cite.authors)}}},\n"
            entry += f"  journal = {{{cite.venue}}},\n"
            entry += f"  year = {{{cite.year}}},\n"
            if cite.doi:
                entry += f"  doi = {{{cite.doi}}},\n"
            entry += f"  url = {{{cite.url}}}\n"
            entry += "}"
            entries.append(entry)
        return "\n\n".join(entries)
    
    def save(self, output_dir: Path):
        """Save paper in multiple formats."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaTeX
        (output_dir / "main.tex").write_text(self.to_latex())
        
        # Markdown
        (output_dir / "main.md").write_text(self.to_markdown())
        
        # BibTeX
        (output_dir / "references.bib").write_text(self.to_bibtex())
        
        # JSON metadata
        meta = {
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "sections": [{"name": s.name, "citations": s.citations, "figures": s.figures} for s in self.sections],
            "citations": [{"key": c.key, "title": c.title, "authors": c.authors, "venue": c.venue, "year": c.year, "url": c.url, "doi": c.doi} for c in self.citations],
            "figures": [{"caption": f.caption, "image_path": f.image_path, "source_paper": f.source_paper, "page_num": f.page_num} for f in self.figures],
            "keywords": self.keywords,
            "generated": datetime.now().isoformat(),
        }
        (output_dir / "paper_metadata.json").write_text(json.dumps(meta, indent=2))
        
        console.print(f"[green]Paper saved to {output_dir}[/green]")

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}', '\\': r'\textbackslash{}',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text


class PaperGenerator:
    def __init__(self, llm=None):
        self.llm = llm
        self.citation_counter = 0
        self.figure_counter = 0
    
    def _generate_citation_key(self, title: str, year: int) -> str:
        """Generate citation key from title and year."""
        words = re.findall(r'\w+', title.lower())
        key_words = [w for w in words if len(w) > 3][:3]
        return f"{''.join(key_words)}{year}"
    
    def _extract_citations_from_sources(self, sources: List[Dict]) -> List[Citation]:
        """Convert source dicts to Citation objects."""
        citations = []
        for i, src in enumerate(sources):
            key = src.get("citation_key") or self._generate_citation_key(src.get("title", ""), 
                src.get("year", datetime.now().year))
            citations.append(Citation(
                key=key,
                title=src.get("title", "Unknown"),
                authors=src.get("authors", ["Unknown"]),
                venue=src.get("venue", "arXiv"),
                year=src.get("year", datetime.now().year),
                url=src.get("url", ""),
                doi=src.get("doi"),
            ))
        return citations
    
    def _fetch_figures_from_paper(self, paper_url: str, paper_title: str) -> List[Figure]:
        """Try to fetch figures from a paper (PDF or HTML)."""
        figures = []
        
        if not HAS_PYMUPDF:
            return figures
        
        try:
            # Try to get PDF
            pdf_url = paper_url
            if "arxiv.org/abs/" in paper_url:
                pdf_url = paper_url.replace("/abs/", "/pdf/") + ".pdf"
            elif "arxiv.org/pdf/" not in paper_url:
                # Try to find PDF link
                pass
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                resp = httpx.get(pdf_url, timeout=30, follow_redirects=True)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
            
            # Extract figures from PDF
            doc = fitz.open(tmp_path)
            for page_num in range(min(len(doc), 10)):  # Limit pages
                page = doc[page_num]
                images = page.get_images(full=True)
                for img_idx, img in enumerate(images):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    if base_image["width"] > 200 and base_image["height"] > 150:  # Filter small images
                        img_data = base_image["image"]
                        img_ext = base_image["ext"]
                        
                        # Save figure
                        fig_filename = f"fig_{self.figure_counter}_{page_num}_{img_idx}.{img_ext}"
                        self.figure_counter += 1
                        
                        figures.append(Figure(
                            caption=f"Figure from {paper_title} (page {page_num+1})",
                            image_path=fig_filename,
                            source_paper=paper_title,
                            page_num=page_num + 1
                        ))
            doc.close()
            os.unlink(tmp_path)
            
        except Exception as e:
            console.print(f"[yellow]Figure extraction failed for {paper_title}:[/yellow] {e}")
        
        return figures
    
    def generate_from_research(
        self,
        topic: str,
        research_report: str,
        sources: List[Dict],
        output_dir: Path
    ) -> ResearchPaper:
        """Generate a paper from deep research results."""
        
        # Extract citations
        citations = self._extract_citations_from_sources(sources)
        
        # Extract key information from report
        sections = self._parse_report_into_sections(research_report, citations)
        
        # Fetch figures from top sources
        figures = []
        for src in sources[:3]:  # Top 3 sources
            if src.get("url"):
                figs = self._fetch_figures_from_paper(src["url"], src.get("title", ""))
                for fig in figs:
                    figures.append(fig)
        
        # Assign figures to sections (simple heuristic: distribute evenly)
        for i, section in enumerate(sections):
            if figures and i < len(figures):
                section.figures = [i]
        
        # Generate paper
        paper = ResearchPaper(
            title=self._generate_title(topic, research_report),
            authors=["Research Agent"],
            abstract=self._generate_abstract(research_report),
            sections=sections,
            citations=citations,
            figures=figures,
            keywords=self._extract_keywords(topic, research_report)
        )
        
        # Save
        paper.save(output_dir)
        
        return paper
    
    def _generate_title(self, topic: str, report: str) -> str:
        if self.llm:
            prompt = f"Generate a concise academic paper title (max 12 words) for research on: {topic}\nReport summary: {report[:1000]}"
            return self.llm.invoke(prompt).content.strip()
        return f"An Investigation of {topic.title()}"
    
    def _generate_abstract(self, report: str) -> str:
        if self.llm:
            prompt = f"Write a concise academic abstract (150-250 words) summarizing this research:\n{report[:3000]}"
            return self.llm.invoke(prompt).content.strip()
        return report[:500] + "..."
    
    def _extract_keywords(self, topic: str, report: str) -> List[str]:
        words = re.findall(r'\b[a-z]{4,}\b', (topic + " " + report[:2000]).lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        # Filter common words
        stopwords = {"this", "that", "with", "from", "have", "been", "were", "their", "which", "would", "there", "could", "other", "than", "about", "these", "into", "only", "also", "such", "more", "very", "some", "time", "just", "like", "then", "over", "after", "before", "between", "through", "during", "without", "under", "within", "among"}
        filtered = [(w, c) for w, c in freq.items() if w not in stopwords]
        filtered.sort(key=lambda x: -x[1])
        return [w for w, _ in filtered[:8]]
    
    def _parse_report_into_sections(self, report: str, citations: List[Citation]) -> List[PaperSection]:
        """Parse the research report into structured sections."""
        sections = []
        
        # Simple section detection
        section_patterns = [
            (r"executive summary", "Executive Summary"),
            (r"key findings?", "Key Findings"),
            (r"detailed analysis", "Detailed Analysis"),
            (r"methodology|methods?", "Methodology"),
            (r"results?|findings?", "Results"),
            (r"discussion", "Discussion"),
            (r"conclusions?|concluding", "Conclusions"),
            (r"future work|limitations?", "Future Work & Limitations"),
            (r"related work|background", "Related Work"),
        ]
        
        # Split report into chunks
        paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
        
        current_section = None
        current_content = []
        
        for para in paragraphs:
            # Check if this paragraph is a section header
            matched_section = None
            for pattern, name in section_patterns:
                if re.search(pattern, para, re.IGNORECASE) and len(para) < 100:
                    matched_section = name
                    break
            
            if matched_section:
                # Save previous section
                if current_section:
                    sections.append(PaperSection(
                        name=current_section,
                        content="\n\n".join(current_content),
                        citations=[]
                    ))
                current_section = matched_section
                current_content = []
            else:
                current_content.append(para)
        
        # Save last section
        if current_section:
            sections.append(PaperSection(
                name=current_section,
                content="\n\n".join(current_content),
                citations=[]
            ))
        
        # Default sections if none found
        if not sections:
            sections = [
                PaperSection(name="Introduction", content=report[:1500]),
                PaperSection(name="Analysis", content=report[1500:3000]),
                PaperSection(name="Conclusions", content=report[3000:]),
            ]
        
        # Assign citations to sections (simple heuristic)
        for section in sections:
            # Find citation numbers mentioned in content
            cite_nums = re.findall(r'\[(\d+)\]', section.content)
            section.citations = [int(n)-1 for n in cite_nums if int(n)-1 < len(citations)]
        
        return sections


def generate_research_paper(
    topic: str,
    research_report: str,
    sources: List[Dict],
    output_dir: str = "./paper_output"
) -> ResearchPaper:
    """Main entry point for paper generation."""
    output_path = Path(output_dir)
    generator = PaperGenerator()
    return generator.generate_from_research(topic, research_report, sources, output_path)