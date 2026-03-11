from __future__ import annotations

import logging
import re
import subprocess
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path

from paper_monitor.llm import LLMClient
from paper_monitor.models import EnrichmentConfig, PaperRecord, RunStats, Settings
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary
from paper_monitor.utils import (
    clean_extracted_text,
    command_exists,
    ensure_directory,
    now_iso,
    shorten,
    stable_hash,
)


LOGGER = logging.getLogger(__name__)

STREAM_RE = re.compile(rb"stream\r?\n(.*?)\r?\nendstream", re.S)
TEXT_LITERAL_RE = re.compile(r"\(([^()]*)\)")


@dataclass(slots=True)
class DocumentArtifacts:
    pdf_local_path: str
    pdf_status: str
    pdf_downloaded_at: str | None
    fulltext_txt_path: str
    fulltext_excerpt: str
    fulltext_status: str
    page_count: int | None
    was_downloaded: bool
    was_extracted: bool


class DocumentProcessor:
    def __init__(self, config: EnrichmentConfig, user_agent: str, timezone_name: str) -> None:
        self.config = config
        self.user_agent = user_agent
        self.timezone_name = timezone_name
        self.has_pdftotext = bool(config.use_pdftotext and command_exists("pdftotext"))
        self.has_pdfinfo = command_exists("pdfinfo")
        ensure_directory(config.pdf_dir)
        ensure_directory(config.text_dir)

    def can_try_pdf(self, paper: PaperRecord) -> bool:
        return bool(self.resolve_pdf_url(paper))

    def resolve_pdf_url(self, paper: PaperRecord) -> str:
        if self._looks_like_pdf_url(paper.pdf_url):
            return paper.pdf_url
        if paper.arxiv_id:
            return f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"
        if "/abs/" in paper.primary_url and "arxiv.org" in paper.primary_url:
            return paper.primary_url.replace("/abs/", "/pdf/") + ".pdf"
        return ""

    def enrich(self, paper: PaperRecord, force: bool = False) -> DocumentArtifacts:
        pdf_url = self.resolve_pdf_url(paper)
        if not pdf_url:
            return DocumentArtifacts(
                pdf_local_path=paper.pdf_local_path,
                pdf_status="no-pdf",
                pdf_downloaded_at=paper.pdf_downloaded_at,
                fulltext_txt_path=paper.fulltext_txt_path,
                fulltext_excerpt=paper.fulltext_excerpt,
                fulltext_status=paper.fulltext_status or "empty",
                page_count=paper.page_count,
                was_downloaded=False,
                was_extracted=False,
            )

        pdf_path = self.config.pdf_dir / f"{self._paper_slug(paper)}.pdf"
        text_path = self.config.text_dir / f"{self._paper_slug(paper)}.txt"
        downloaded_at = paper.pdf_downloaded_at
        was_downloaded = False

        if force or self.config.redownload_existing or not pdf_path.exists():
            downloaded_at = now_iso(self.timezone_name)
            self._download_pdf(pdf_url, pdf_path)
            was_downloaded = True

        page_count = self._extract_page_count(pdf_path) if pdf_path.exists() else paper.page_count

        fulltext = ""
        was_extracted = False
        if force or not text_path.exists() or self.config.redownload_existing:
            fulltext = self._extract_text(pdf_path)
            if fulltext:
                text_path.write_text(fulltext + "\n", encoding="utf-8")
                was_extracted = True
        elif text_path.exists():
            fulltext = text_path.read_text(encoding="utf-8", errors="replace")

        fulltext = clean_extracted_text(fulltext)
        return DocumentArtifacts(
            pdf_local_path=str(pdf_path) if pdf_path.exists() else "",
            pdf_status="downloaded" if pdf_path.exists() else "failed",
            pdf_downloaded_at=downloaded_at,
            fulltext_txt_path=str(text_path) if text_path.exists() else "",
            fulltext_excerpt=shorten(fulltext, self.config.excerpt_chars) if fulltext else "",
            fulltext_status="extracted" if fulltext else "failed",
            page_count=page_count,
            was_downloaded=was_downloaded,
            was_extracted=was_extracted,
        )

    def _paper_slug(self, paper: PaperRecord) -> str:
        if paper.arxiv_id:
            return paper.arxiv_id.replace("/", "_")
        if paper.doi:
            return stable_hash(paper.doi)
        return f"paper-{paper.id}-{stable_hash(paper.title)[:10]}"

    def _looks_like_pdf_url(self, url: str) -> bool:
        lowered = url.lower().strip()
        if not lowered:
            return False
        if lowered.endswith(".pdf"):
            return True
        return "/pdf/" in lowered

    def _download_pdf(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=self.config.download_timeout_seconds) as response:
            destination.write_bytes(response.read())

    def _extract_page_count(self, pdf_path: Path) -> int | None:
        if not self.has_pdfinfo:
            return None
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                value = line.split(":", 1)[1].strip()
                if value.isdigit():
                    return int(value)
        return None

    def _extract_text(self, pdf_path: Path) -> str:
        if self.has_pdftotext:
            result = subprocess.run(
                ["pdftotext", "-layout", "-nopgbrk", str(pdf_path), "-"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return clean_extracted_text(result.stdout)
        return self._fallback_extract_text(pdf_path)

    def _fallback_extract_text(self, pdf_path: Path) -> str:
        data = pdf_path.read_bytes()
        chunks: list[str] = []
        for stream in STREAM_RE.findall(data):
            decoded = stream
            for _ in range(2):
                try:
                    decoded = zlib.decompress(decoded)
                    break
                except zlib.error:
                    pass
            text = decoded.decode("latin-1", errors="ignore")
            literals = TEXT_LITERAL_RE.findall(text)
            if literals:
                chunks.extend(literals)
        return clean_extracted_text(" ".join(chunks))


class EnrichmentPipeline:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        document_processor: DocumentProcessor | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        user_agent = settings.arxiv.user_agent or settings.dblp.user_agent
        self.document_processor = document_processor or DocumentProcessor(
            settings.enrichment,
            user_agent=user_agent,
            timezone_name=settings.timezone,
        )
        self.llm_client = llm_client or LLMClient(settings.llm)

    def run(
        self,
        *,
        limit: int | None = None,
        topic_ids: list[str] | None = None,
        classifications: list[str] | None = None,
        force: bool = False,
        use_llm: bool | None = None,
    ) -> RunStats:
        stats = RunStats()
        if not self.settings.enrichment.enabled and not force:
            return stats

        selected_classifications = classifications or self.settings.enrichment.process_classifications
        candidates = self.db.fetch_enrichment_candidates(
            limit=limit or self.settings.enrichment.max_papers_per_run,
            classifications=selected_classifications,
            topic_ids=topic_ids,
        )
        use_llm_effective = self.llm_client.enabled if use_llm is None else (use_llm and self.llm_client.enabled)

        for paper in candidates:
            if self._should_skip(paper, force=force, use_llm=use_llm_effective):
                stats.skipped += 1
                continue
            try:
                self._enrich_paper(paper, stats, force=force, use_llm=use_llm_effective)
            except (OSError, subprocess.SubprocessError, urllib.error.URLError) as exc:
                LOGGER.warning("enrichment failed for paper_id=%s: %s", paper.id, exc)
                stats.errors.append(f"paper:{paper.id}:{exc}")
        return stats

    def _should_skip(self, paper: PaperRecord, *, force: bool, use_llm: bool) -> bool:
        if force:
            return False
        needs_pdf = self.document_processor.can_try_pdf(paper) and paper.fulltext_status != "extracted"
        needs_pdf_state_refresh = paper.pdf_status == "pending"
        needs_llm = use_llm and not paper.summary_basis.startswith("llm+")
        needs_offline_refresh = (
            not use_llm
            and paper.fulltext_status == "extracted"
            and paper.summary_basis not in {"fulltext+metadata", "llm+fulltext+metadata"}
        )
        needs_initial_summary = not paper.summary_text
        return not (needs_pdf or needs_pdf_state_refresh or needs_llm or needs_offline_refresh or needs_initial_summary)

    def _enrich_paper(self, paper: PaperRecord, stats: RunStats, *, force: bool, use_llm: bool) -> None:
        if force or paper.fulltext_status != "extracted" or paper.pdf_status == "pending":
            artifacts = self.document_processor.enrich(paper, force=force)
            self.db.update_paper_assets(
                paper.id,
                pdf_local_path=artifacts.pdf_local_path,
                pdf_status=artifacts.pdf_status,
                pdf_downloaded_at=artifacts.pdf_downloaded_at,
                fulltext_txt_path=artifacts.fulltext_txt_path,
                fulltext_excerpt=artifacts.fulltext_excerpt,
                fulltext_status=artifacts.fulltext_status,
                page_count=artifacts.page_count,
            )
            if artifacts.was_downloaded:
                stats.downloaded_pdfs += 1
            if artifacts.was_extracted:
                stats.extracted_texts += 1
            paper = self.db.get_paper(paper.id)

        evaluations = self.db.fetch_paper_evaluations(paper.id)

        llm_result = self.llm_client.generate_summary(paper, evaluations) if use_llm else None
        if llm_result is not None:
            self.db.update_paper_analysis(
                paper.id,
                llm_result.summary_text,
                llm_result.summary_basis,
                llm_result.tags,
                llm_summary=llm_result.structured,
            )
            stats.llm_summaries += 1
        else:
            summary_text, basis, tags = build_paper_summary(paper, evaluations)
            self.db.update_paper_analysis(paper.id, summary_text, basis, tags)

        stats.enriched += 1
