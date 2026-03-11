from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_monitor.config import load_settings
from paper_monitor.enrichment import DocumentArtifacts, EnrichmentPipeline
from paper_monitor.models import LLMResult
from paper_monitor.models import PaperCandidate, RunStats
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_report
from paper_monitor.storage import Database


class MonitorPipelineTest(unittest.TestCase):
    def test_pipeline_scoring_and_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            matrix_candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2401.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Matrix-Free Finite Element Operator Evaluation on GPUs",
                abstract=(
                    "We present a matrix-free finite element operator evaluation strategy "
                    "with partial assembly and sum factorization on GPUs."
                ),
                authors=["Alice", "Bob"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2401.00001",
                pdf_url="https://arxiv.org/pdf/2401.00001.pdf",
                doi="",
                arxiv_id="2401.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA", "cs.MS"],
                raw={"fixture": True},
            )
            ai_candidate = PaperCandidate(
                source_name="dblp",
                source_paper_id="conf/example/flashattention",
                query_text="FlashAttention",
                title="Kernel Fusion for Transformer Inference with FlashAttention and Triton",
                abstract=(
                    "This paper studies kernel fusion for transformer inference and describes "
                    "a Triton-based code generation path for attention kernels."
                ),
                authors=["Carol"],
                published_at="2026-03-10",
                updated_at="2026-03-10",
                primary_url="https://dblp.org/rec/conf/example/flashattention",
                pdf_url="",
                doi="10.1000/example",
                arxiv_id="",
                venue="MLSys",
                year=2026,
                categories=["Conference"],
                raw={"fixture": True},
            )

            pipeline._process_candidate(matrix_candidate, RunStats())
            pipeline._process_candidate(ai_candidate, RunStats())

            paths = generate_report(db, settings, report_date="2026-03-10", report_type="daily", lookback_days=1)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            html = Path(paths["html"]).read_text(encoding="utf-8")

            self.assertIn("Matrix-Free Finite Element Operator Evaluation on GPUs", markdown)
            self.assertIn("Kernel Fusion for Transformer Inference with FlashAttention and Triton", markdown)
            self.assertIn("有限元分析 Matrix-Free 算法优化", html)
            self.assertIn("AI 算子加速", html)

            db.close()

    def test_enrichment_pipeline_upgrades_summary_with_fulltext_and_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enabled = True
            settings.llm.base_url = "http://example.invalid/v1"
            settings.llm.model = "dummy-model"

            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            candidate = PaperCandidate(
                source_name="arxiv",
                source_paper_id="2501.00001",
                query_text="all:\"matrix-free\" AND all:\"finite element\"",
                title="Matrix-Free Finite Element Preconditioners with Partial Assembly",
                abstract=(
                    "We present matrix-free finite element preconditioners with partial assembly "
                    "and show strong GPU performance."
                ),
                authors=["Alice"],
                published_at="2026-03-10T09:00:00+08:00",
                updated_at="2026-03-10T09:00:00+08:00",
                primary_url="https://arxiv.org/abs/2501.00001",
                pdf_url="https://arxiv.org/pdf/2501.00001.pdf",
                doi="",
                arxiv_id="2501.00001",
                venue="arXiv",
                year=2026,
                categories=["cs.NA"],
                raw={"fixture": True},
            )
            pipeline._process_candidate(candidate, RunStats())

            class FakeDocumentProcessor:
                def can_try_pdf(self, paper):  # noqa: ANN001
                    return True

                def enrich(self, paper, force=False):  # noqa: ANN001, ARG002
                    pdf_path = settings.enrichment.pdf_dir / "2501.00001.pdf"
                    text_path = settings.enrichment.text_dir / "2501.00001.txt"
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)
                    text_path.parent.mkdir(parents=True, exist_ok=True)
                    pdf_path.write_bytes(b"%PDF-1.4\n")
                    text_content = (
                        "This work introduces a matrix-free finite element implementation with "
                        "partial assembly, multigrid preconditioners, and strong GPU scaling."
                    )
                    text_path.write_text(text_content, encoding="utf-8")
                    return DocumentArtifacts(
                        pdf_local_path=str(pdf_path),
                        pdf_status="downloaded",
                        pdf_downloaded_at="2026-03-10T10:00:00+08:00",
                        fulltext_txt_path=str(text_path),
                        fulltext_excerpt=text_content,
                        fulltext_status="extracted",
                        page_count=12,
                        was_downloaded=True,
                        was_extracted=True,
                    )

            class FakeLLMClient:
                enabled = True

                def generate_summary(self, paper, evaluations):  # noqa: ANN001
                    self.last_title = paper.title
                    self.last_scores = [item.score for item in evaluations]
                    return LLMResult(
                        summary_text="问题：高阶有限元矩阵自由算子成本高。方法：结合 partial assembly 与 multigrid。贡献：GPU 性能提升明显。",
                        summary_basis="llm+fulltext+metadata",
                        tags=["matrix-free", "partial assembly", "multigrid"],
                        structured={
                            "summary": "结合全文节选生成的中文总结。",
                            "tags": ["matrix-free", "partial assembly", "multigrid"],
                        },
                    )

            enrichment_pipeline = EnrichmentPipeline(
                settings,
                db,
                document_processor=FakeDocumentProcessor(),
                llm_client=FakeLLMClient(),
            )
            stats = enrichment_pipeline.run(limit=5, force=False, use_llm=True)
            self.assertEqual(stats.enriched, 1)
            self.assertEqual(stats.downloaded_pdfs, 1)
            self.assertEqual(stats.extracted_texts, 1)
            self.assertEqual(stats.llm_summaries, 1)

            paper = db.get_paper(1)
            self.assertEqual(paper.fulltext_status, "extracted")
            self.assertEqual(paper.summary_basis, "llm+fulltext+metadata")
            self.assertIn("partial assembly", paper.summary_text)
            self.assertEqual(paper.page_count, 12)

            paths = generate_report(db, settings, report_date="2026-03-10", report_type="daily", lookback_days=1)
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("全文状态", markdown)
            self.assertIn("llm+fulltext+metadata", markdown)

            db.close()

    def test_report_includes_topic_digest_when_llm_client_is_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            source_config = Path("/home/momo/git_ws/search/config/config.example.json").read_text(encoding="utf-8")
            (root / "config" / "config.json").write_text(source_config, encoding="utf-8")

            settings = load_settings(root / "config" / "config.json")
            settings.llm.enable_topic_digest = True
            db = Database(settings.database_path, settings.timezone)
            db.initialize()
            pipeline = MonitorPipeline(settings, db)

            pipeline._process_candidate(
                PaperCandidate(
                    source_name="arxiv",
                    source_paper_id="2601.00001",
                    query_text="FlashAttention",
                    title="FlashAttention Kernel Fusion for Transformer Training",
                    abstract="We study kernel fusion and Triton code generation for transformer training.",
                    authors=["Alice"],
                    published_at="2026-03-10T09:00:00+08:00",
                    updated_at="2026-03-10T09:00:00+08:00",
                    primary_url="https://arxiv.org/abs/2601.00001",
                    pdf_url="https://arxiv.org/pdf/2601.00001.pdf",
                    doi="",
                    arxiv_id="2601.00001",
                    venue="arXiv",
                    year=2026,
                    categories=["cs.LG"],
                    raw={"fixture": True},
                ),
                RunStats(),
            )

            class FakeTopicDigestClient:
                enabled = True

                def generate_topic_digest(self, topic_name, description, entries):  # noqa: ANN001
                    if topic_name != "AI 算子加速":
                        return None
                    return type(
                        "Digest",
                        (),
                        {
                            "overview": "本窗口主题主要集中在 attention kernel 与 kernel fusion。",
                            "highlights": ["FlashAttention 仍然是主线", "Triton 和 MLIR 协同增多"],
                            "watchlist": ["关注训练侧 kernel autotuning"],
                            "tags": ["flashattention", "kernel fusion"],
                            "structured": {"usage": {"input_tokens": 1200, "output_tokens": 180, "total_tokens": 1380}},
                        },
                    )()

            paths = generate_report(
                db,
                settings,
                report_date="2026-03-10",
                report_type="daily",
                lookback_days=1,
                llm_client=FakeTopicDigestClient(),
                use_llm_topic_digest=True,
            )
            markdown = Path(paths["markdown"]).read_text(encoding="utf-8")
            self.assertIn("LLM 主题概览", markdown)
            self.assertIn("attention kernel 与 kernel fusion", markdown)
            self.assertIn("LLM Token", markdown)

            db.close()


if __name__ == "__main__":
    unittest.main()
