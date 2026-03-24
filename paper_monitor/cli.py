from __future__ import annotations

from pathlib import Path
import argparse
from dataclasses import replace
import logging
import shlex
from collections import Counter

from paper_monitor.config import load_settings, write_default_config
from paper_monitor.enrichment import DocumentProcessor, EnrichmentPipeline
from paper_monitor.fetchers.arxiv import ArxivFetcher, extract_arxiv_id
from paper_monitor.llm import LLMClient
from paper_monitor.llm_registry import build_runtime_variants
from paper_monitor.models import PaperCandidate, PaperLLMSummary, PaperRecord, TopicEvaluation
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.progress import ProgressBar
from paper_monitor.reports import (
    generate_catalog_report,
    generate_comparison_report,
    generate_preview_report,
    generate_paper_reports,
    generate_report,
)
from paper_monitor.scoring import evaluate_candidate_against_topic
from paper_monitor.scheduler import run_daemon
from paper_monitor.storage import Database
from paper_monitor.summarize import build_paper_summary
from paper_monitor.utils import ensure_directory, normalize_title, now_iso, stable_hash, to_day_bounds, today_string


LOGGER = logging.getLogger(__name__)


ENRICH_CHECKPOINT_KEY = "enrich:since"
DEFAULT_RUNTIME_CONFIG = "config/config-poe.json"
DEFAULT_INIT_CONFIG = "config/config.example.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地论文监控工具")
    parser.add_argument("--config", default=DEFAULT_RUNTIME_CONFIG, help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="创建默认配置并初始化目录")
    init_parser.add_argument("--force", action="store_true", help="覆盖现有配置文件")

    fetch_parser = subparsers.add_parser("fetch", help="仅抓取并写入数据库")
    fetch_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])
    fetch_parser.add_argument("--start-year", type=int, help="首次回填时只保留不早于该年份的论文")
    fetch_parser.add_argument("--recent-limit", type=int, help="按时间顺序保留最近多少篇论文")
    fetch_parser.add_argument("--page-size", type=int, help="单次请求分页大小")
    fetch_parser.add_argument("--since-last-run", action="store_true", help="只处理自上次成功抓取后新增的论文")

    enrich_parser = subparsers.add_parser("enrich", help="下载 PDF、抽取全文并刷新摘要")
    enrich_parser.add_argument("--limit", type=int, help="本轮最多增强多少篇论文")
    enrich_parser.add_argument("--topic", action="append", help="仅增强指定 topic id")
    enrich_parser.add_argument("--classification", action="append", choices=["relevant", "maybe"])
    enrich_parser.add_argument("--force", action="store_true", help="忽略缓存，重新下载或重新抽取")
    enrich_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则强制尝试生成 LLM 摘要")
    enrich_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    enrich_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    enrich_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    enrich_parser.add_argument("--paper-id", type=int, action="append", help="仅增强指定一个或多个 paper_id")
    enrich_parser.add_argument("--since-last-run", action="store_true", help="只增强自上次成功增强后新增入库的论文")
    enrich_parser.add_argument("--secondary-on-priority", action="store_true", help="仅让额外模型处理高优先级论文；主模型仍处理全部论文")
    enrich_parser.add_argument("--secondary-top-per-topic", type=int, default=3, help="开启 --secondary-on-priority 后，每个领域送给额外模型二次总结的 Top N，默认 3")
    enrich_parser.add_argument("--secondary-min-score", type=float, default=24.0, help="开启 --secondary-on-priority 后，评分不低于该值的论文会送给额外模型，默认 24")
    enrich_parser.add_argument("--retry-from-variant", help="只处理另一个模型仍缺失或回退的论文，例如 ikun 或 ikun_gpt-5.4")
    enrich_parser.add_argument("--retry-from-status", choices=["incomplete", "fallback", "missing"], default="incomplete", help="和 --retry-from-variant 一起使用：incomplete=缺失或回退，fallback=仅回退，missing=仅缺失")

    report_parser = subparsers.add_parser("report", help="仅生成报告")
    report_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    report_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    report_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    report_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则生成主题级聚合摘要")
    report_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    report_parser.add_argument("--digest-primary-only", action="store_true", help="仅让主配置模型生成领域级聚合摘要；单篇多模型摘要仍会保留")

    paper_report_parser = subparsers.add_parser("paper-report", help="生成单篇论文的 Markdown / HTML / JSON 报告")
    paper_report_parser.add_argument("--paper-id", type=int, action="append", help="指定一个或多个 paper_id")
    paper_report_parser.add_argument("--date", help="若不指定 --paper-id，则按时间窗口导出对应论文")
    paper_report_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    paper_report_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    paper_report_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")

    catalog_parser = subparsers.add_parser("catalog-report", help="按当前数据库生成全库总览报告")
    catalog_parser.add_argument("--topic", action="append", help="仅导出指定 topic id")
    catalog_parser.add_argument("--classification", action="append", choices=["relevant", "maybe"])
    catalog_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则生成领域级聚合概览")
    catalog_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    catalog_parser.add_argument("--digest-primary-only", action="store_true", help="仅让主配置模型生成领域级聚合概览；单篇多模型摘要仍会保留")

    paper_add_parser = subparsers.add_parser("paper-add", help="手动向数据库添加一篇论文并绑定到一个或多个领域")
    paper_add_parser.add_argument("--topic", action="append", required=True, help="目标 topic id，可重复传入")
    paper_add_parser.add_argument("--title", required=True, help="论文标题")
    paper_add_parser.add_argument("--abstract", default="", help="论文摘要")
    paper_add_parser.add_argument("--author", action="append", help="作者，可重复传入")
    paper_add_parser.add_argument("--published-at", help="发布时间，支持 YYYY-MM-DD 或 ISO 时间")
    paper_add_parser.add_argument("--updated-at", help="更新时间，支持 YYYY-MM-DD 或 ISO 时间")
    paper_add_parser.add_argument("--primary-url", default="", help="论文主页或详情链接")
    paper_add_parser.add_argument("--pdf-url", default="", help="PDF 下载链接")
    paper_add_parser.add_argument("--doi", default="", help="DOI")
    paper_add_parser.add_argument("--arxiv-id", default="", help="arXiv ID")
    paper_add_parser.add_argument("--venue", default="", help="期刊 / 会议 / 来源名称")
    paper_add_parser.add_argument("--year", type=int, help="年份")
    paper_add_parser.add_argument("--category", action="append", help="分类标签，可重复传入")
    paper_add_parser.add_argument("--classification", default="relevant", choices=["relevant", "maybe"])

    paper_set_pdf_parser = subparsers.add_parser("paper-set-pdf", help="为已有论文设置或替换 PDF 源，并重置 PDF / 全文缓存")
    paper_set_pdf_parser.add_argument("--paper-id", type=int, required=True, help="要更新的 paper_id")
    paper_set_pdf_parser.add_argument("--pdf-url", required=True, help="新的 PDF 下载链接，或 file:/// 开头的本地 PDF 绝对路径")

    paper_delete_parser = subparsers.add_parser("paper-delete", help="按 paper_id 从数据库删除论文")
    paper_delete_parser.add_argument("--paper-id", type=int, action="append", required=True, help="一个或多个 paper_id")

    paper_find_parser = subparsers.add_parser("paper-find", help="按标题、URL、DOI、arXiv ID 或领域查找论文，并显示 paper_id")
    paper_find_parser.add_argument("--title", help="按标题关键字模糊查找")
    paper_find_parser.add_argument("--url", help="按 primary_url / pdf_url / source_url 模糊查找")
    paper_find_parser.add_argument("--doi", help="按 DOI 精确查找")
    paper_find_parser.add_argument("--arxiv-id", help="按 arXiv ID 精确查找")
    paper_find_parser.add_argument("--topic", action="append", help="仅在指定 topic id 内查找，可重复传入")
    paper_find_parser.add_argument("--no-pdf", action="store_true", help="仅列出当前没有本地 PDF 的论文")
    paper_find_parser.add_argument("--show-summary", action="store_true", help="同时显示当前默认摘要内容")
    paper_find_parser.add_argument("--limit", type=int, default=20, help="最多显示多少条，默认 20")

    preview_parser = subparsers.add_parser("paper-preview", help="仅对一篇论文做评分、下载 PDF、总结和导出，不写入数据库")
    preview_parser.add_argument("--topic", action="append", help="仅按指定 topic id 评分；默认评估全部领域")
    preview_parser.add_argument("--title", help="论文标题；如果只给 arXiv 链接或 arXiv ID，可以自动补全")
    preview_parser.add_argument("--abstract", default="", help="论文摘要")
    preview_parser.add_argument("--author", action="append", help="作者，可重复传入")
    preview_parser.add_argument("--published-at", help="发布时间，支持 YYYY-MM-DD 或 ISO 时间")
    preview_parser.add_argument("--updated-at", help="更新时间，支持 YYYY-MM-DD 或 ISO 时间")
    preview_parser.add_argument("--primary-url", default="", help="论文主页或详情链接")
    preview_parser.add_argument("--pdf-url", default="", help="PDF 下载链接")
    preview_parser.add_argument("--doi", default="", help="DOI")
    preview_parser.add_argument("--arxiv-id", default="", help="arXiv ID")
    preview_parser.add_argument("--venue", default="", help="期刊 / 会议 / 来源名称")
    preview_parser.add_argument("--year", type=int, help="年份")
    preview_parser.add_argument("--category", action="append", help="分类标签，可重复传入")
    preview_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则生成单篇多模型总结")
    preview_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    preview_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    preview_parser.add_argument("--force", action="store_true", help="忽略已有本地文件，重新下载或重新抽取")

    compare_parser = subparsers.add_parser("compare-report", help="使用两套配置生成并排对比报告")
    compare_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    compare_parser.add_argument("--type", default="weekly", choices=["daily", "weekly"])
    compare_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用左侧配置中的 report.lookback_days")
    compare_parser.add_argument("--left-config", default="config/config-poe.json", help="左侧配置文件路径")
    compare_parser.add_argument("--right-config", default="config/config-ikun.json", help="右侧配置文件路径")

    run_once_parser = subparsers.add_parser("run-once", help="抓取后立即生成报告")
    run_once_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    run_once_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    run_once_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    run_once_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])
    run_once_parser.add_argument("--enrich", action="store_true", help="抓取后执行全文增强")
    run_once_parser.add_argument("--enrich-limit", type=int, help="本轮增强的论文数量上限")
    run_once_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")
    run_once_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    run_once_parser.add_argument("--digest-primary-only", action="store_true", help="仅让主配置模型生成领域级聚合摘要；单篇多模型摘要仍会保留")
    run_once_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    run_once_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    run_once_parser.add_argument("--start-year", type=int, help="首次回填时只保留不早于该年份的论文")
    run_once_parser.add_argument("--recent-limit", type=int, help="按时间顺序保留最近多少篇论文")
    run_once_parser.add_argument("--page-size", type=int, help="单次请求分页大小")
    run_once_parser.add_argument("--since-last-run", action="store_true", help="只处理自上次成功抓取后新增的论文")
    run_once_parser.add_argument("--secondary-on-priority", action="store_true", help="仅让额外模型处理高优先级论文；主模型仍处理全部论文")
    run_once_parser.add_argument("--secondary-top-per-topic", type=int, default=3, help="开启 --secondary-on-priority 后，每个领域送给额外模型二次总结的 Top N，默认 3")
    run_once_parser.add_argument("--secondary-min-score", type=float, default=24.0, help="开启 --secondary-on-priority 后，评分不低于该值的论文会送给额外模型，默认 24")
    run_once_parser.add_argument("--retry-from-variant", help="只让当前模型补跑另一个模型仍缺失或回退的论文，例如 ikun 或 ikun_gpt-5.4")
    run_once_parser.add_argument("--retry-from-status", choices=["incomplete", "fallback", "missing"], default="incomplete", help="和 --retry-from-variant 一起使用：incomplete=缺失或回退，fallback=仅回退，missing=仅缺失")

    daemon_parser = subparsers.add_parser("daemon", help="持续轮询抓取，并覆盖更新当天报告")
    daemon_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    daemon_parser.add_argument("--loops", type=int, help="仅运行指定轮次，便于测试")
    daemon_parser.add_argument("--enrich", action="store_true", help="每轮抓取后执行全文增强")
    daemon_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")
    daemon_parser.add_argument("--extra-config", action="append", help="额外加载一份配置文件中的 LLM 变体")
    daemon_parser.add_argument("--digest-primary-only", action="store_true", help="仅让主配置模型生成领域级聚合摘要；单篇多模型摘要仍会保留")
    daemon_parser.add_argument("--skip-pdf", action="store_true", help="跳过 PDF 下载与全文抽取，只基于标题/摘要生成总结")
    daemon_parser.add_argument("--workers", type=int, default=1, help="增强阶段并发 worker 数，默认 1")
    daemon_parser.add_argument("--since-last-run", action="store_true", help="每轮仅处理自上次成功抓取后新增的论文")
    daemon_parser.add_argument("--secondary-on-priority", action="store_true", help="仅让额外模型处理高优先级论文；主模型仍处理全部论文")
    daemon_parser.add_argument("--secondary-top-per-topic", type=int, default=3, help="开启 --secondary-on-priority 后，每个领域送给额外模型二次总结的 Top N，默认 3")
    daemon_parser.add_argument("--secondary-min-score", type=float, default=24.0, help="开启 --secondary-on-priority 后，评分不低于该值的论文会送给额外模型，默认 24")
    daemon_parser.add_argument("--retry-from-variant", help="只让当前模型补跑另一个模型仍缺失或回退的论文，例如 ikun 或 ikun_gpt-5.4")
    daemon_parser.add_argument("--retry-from-status", choices=["incomplete", "fallback", "missing"], default="incomplete", help="和 --retry-from-variant 一起使用：incomplete=缺失或回退，fallback=仅回退，missing=仅缺失")

    return parser


def _configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _open_database(config_path: str) -> tuple[Database, MonitorPipeline]:
    settings = load_settings(config_path)
    ensure_directory(settings.report_dir)
    ensure_directory(settings.export_dir)
    ensure_directory(settings.database_path.parent)
    ensure_directory(settings.enrichment.pdf_dir)
    ensure_directory(settings.enrichment.text_dir)
    db = Database(settings.database_path, settings.timezone)
    db.initialize()
    return db, MonitorPipeline(settings, db)


def _select_topics(settings, topic_ids: list[str]):  # noqa: ANN001
    topic_map = {topic.id: topic for topic in settings.topics}
    missing = [topic_id for topic_id in topic_ids if topic_id not in topic_map]
    if missing:
        raise ValueError(f"未知 topic id: {', '.join(missing)}")
    return [topic_map[topic_id] for topic_id in topic_ids]


def _build_preview_candidate(args) -> PaperCandidate:  # noqa: ANN001
    source_key = args.pdf_url or args.primary_url or args.doi or args.arxiv_id or args.title
    return PaperCandidate(
        source_name="preview",
        source_paper_id=f"preview-{stable_hash(source_key)}",
        query_text="preview",
        title=args.title,
        abstract=args.abstract or "",
        authors=args.author or [],
        published_at=args.published_at,
        updated_at=args.updated_at,
        primary_url=args.primary_url or args.pdf_url or "",
        pdf_url=args.pdf_url or "",
        doi=args.doi or "",
        arxiv_id=args.arxiv_id or "",
        venue=args.venue or "preview",
        year=args.year,
        categories=args.category or [],
        raw={"preview": True, "added_via": "paper-preview"},
    )


def _autofill_preview_candidate(settings, args):  # noqa: ANN001
    explicit_arxiv = args.arxiv_id or extract_arxiv_id(args.primary_url or "") or extract_arxiv_id(args.pdf_url or "")
    if not explicit_arxiv:
        if args.title:
            return _build_preview_candidate(args)
        raise ValueError("paper-preview 至少需要提供 --title，或提供可识别的 arXiv 链接 / arXiv ID。")

    fetcher = ArxivFetcher(settings.arxiv)
    fetched = fetcher.fetch_single(explicit_arxiv)
    if fetched is None and not args.title:
        raise ValueError(f"无法从 arXiv 自动补全元数据: {explicit_arxiv}")

    if fetched is None:
        return _build_preview_candidate(args)

    return PaperCandidate(
        source_name="preview",
        source_paper_id=f"preview-{stable_hash(args.pdf_url or args.primary_url or explicit_arxiv or fetched.source_paper_id)}",
        query_text="preview",
        title=args.title or fetched.title,
        abstract=args.abstract or fetched.abstract,
        authors=args.author or fetched.authors,
        published_at=args.published_at or fetched.published_at,
        updated_at=args.updated_at or fetched.updated_at,
        primary_url=args.primary_url or fetched.primary_url or args.pdf_url or "",
        pdf_url=args.pdf_url or fetched.pdf_url or "",
        doi=args.doi or fetched.doi,
        arxiv_id=args.arxiv_id or fetched.arxiv_id,
        venue=args.venue or fetched.venue or "preview",
        year=args.year or fetched.year,
        categories=args.category or fetched.categories,
        raw={
            "preview": True,
            "added_via": "paper-preview",
            "autofilled_from_arxiv": bool(fetched),
            "arxiv_source_id": fetched.source_paper_id,
            "arxiv_query": fetched.query_text,
        },
    )


def _build_preview_paper(candidate: PaperCandidate, timezone_name: str) -> PaperRecord:
    preview_key = candidate.doi or candidate.arxiv_id or candidate.pdf_url or candidate.primary_url or candidate.title
    preview_id = int(stable_hash(preview_key)[:8], 16)
    created_at = now_iso(timezone_name)
    return PaperRecord(
        id=preview_id,
        title=candidate.title,
        title_norm=normalize_title(candidate.title),
        abstract=candidate.abstract or "",
        authors=candidate.authors,
        published_at=candidate.published_at,
        updated_at=candidate.updated_at,
        primary_url=candidate.primary_url,
        pdf_url=candidate.pdf_url,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        venue=candidate.venue,
        year=candidate.year,
        categories=candidate.categories,
        summary_text="",
        summary_basis="metadata-only",
        tags=[],
        pdf_local_path="",
        pdf_status="pending",
        pdf_downloaded_at=None,
        fulltext_txt_path="",
        fulltext_excerpt="",
        fulltext_status="empty",
        page_count=None,
        llm_summary={},
        analysis_updated_at=None,
        source_first="preview",
        created_at=created_at,
        last_seen_at=created_at,
        metadata=dict(candidate.raw),
    )


def _preview_paper_add_suggestion(args, evaluations: list[TopicEvaluation]) -> dict:  # noqa: ANN001
    recommended_topics = [
        item.topic_id
        for item in evaluations
        if item.classification in {"relevant", "maybe"}
    ]
    if not recommended_topics:
        return {
            "recommended_topic_ids": [],
            "command": "",
            "note": "当前没有推荐入库领域，请手动确认 topic 后再执行 paper-add。",
        }
    command_parts = ["python3", "main.py", "--config", args.config, "paper-add"]
    for topic_id in recommended_topics:
        command_parts.extend(["--topic", topic_id])
    command_parts.extend(["--title", args.title])
    if args.abstract:
        command_parts.extend(["--abstract", args.abstract])
    for author in args.author or []:
        command_parts.extend(["--author", author])
    if args.published_at:
        command_parts.extend(["--published-at", args.published_at])
    if args.updated_at:
        command_parts.extend(["--updated-at", args.updated_at])
    if args.primary_url:
        command_parts.extend(["--primary-url", args.primary_url])
    if args.pdf_url:
        command_parts.extend(["--pdf-url", args.pdf_url])
    if args.doi:
        command_parts.extend(["--doi", args.doi])
    if args.arxiv_id:
        command_parts.extend(["--arxiv-id", args.arxiv_id])
    if args.venue:
        command_parts.extend(["--venue", args.venue])
    if args.year:
        command_parts.extend(["--year", str(args.year)])
    for category in args.category or []:
        command_parts.extend(["--category", category])
    return {
        "recommended_topic_ids": recommended_topics,
        "command": " ".join(shlex.quote(part) for part in command_parts),
    }


def _run_paper_preview(args) -> int:  # noqa: ANN001
    settings = load_settings(args.config)
    ensure_directory(settings.report_dir)
    ensure_directory(settings.export_dir)
    ensure_directory(settings.enrichment.pdf_dir)
    ensure_directory(settings.enrichment.text_dir)

    extra_configs = getattr(args, "extra_config", None) or []
    runtime_variants = build_runtime_variants(settings, extra_configs)
    enabled_variants = [variant for variant in runtime_variants if variant.client.enabled]
    if args.with_llm and not enabled_variants:
        LOGGER.warning(
            "检测到 --with-llm，但当前没有可用的 LLM 变体。请检查 API key 环境变量和额外配置。"
        )
    if args.skip_pdf and args.with_llm:
        LOGGER.warning("当前使用了 --skip-pdf，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")

    candidate = _autofill_preview_candidate(settings, args)
    paper = _build_preview_paper(candidate, settings.timezone)
    topics = _select_topics(settings, args.topic) if args.topic else settings.topics
    evaluations = sorted(
        [evaluate_candidate_against_topic(candidate, topic) for topic in topics],
        key=lambda item: (item.score, item.topic_name),
        reverse=True,
    )

    document_processor = DocumentProcessor(
        settings.enrichment,
        user_agent=settings.arxiv.user_agent or settings.dblp.user_agent,
        timezone_name=settings.timezone,
    )
    if not args.skip_pdf:
        artifacts = document_processor.enrich(paper, force=args.force)
        paper = replace(
            paper,
            pdf_local_path=artifacts.pdf_local_path,
            pdf_status=artifacts.pdf_status,
            pdf_downloaded_at=artifacts.pdf_downloaded_at,
            fulltext_txt_path=artifacts.fulltext_txt_path,
            fulltext_excerpt=artifacts.fulltext_excerpt,
            fulltext_status=artifacts.fulltext_status,
            page_count=artifacts.page_count,
        )

    llm_summaries: list[PaperLLMSummary] = []
    llm_results: list[tuple[LLMRuntimeVariant, object]] = []
    if args.with_llm:
        for variant in enabled_variants:
            llm_result = variant.client.generate_summary(paper, evaluations)
            if llm_result is None:
                continue
            llm_results.append((variant, llm_result))
            usage = {}
            if isinstance(llm_result.structured, dict):
                usage = llm_result.structured.get("usage", {})
            llm_summaries.append(
                PaperLLMSummary(
                    paper_id=paper.id,
                    variant_id=variant.variant_id,
                    variant_label=variant.label,
                    provider=variant.provider,
                    base_url=variant.base_url,
                    model=variant.model,
                    summary_text=llm_result.summary_text,
                    summary_basis=llm_result.summary_basis,
                    tags=llm_result.tags,
                    structured=llm_result.structured,
                    usage=usage if isinstance(usage, dict) else {},
                    created_at=paper.created_at,
                    updated_at=paper.created_at,
                )
            )

    if llm_results:
        _, primary_llm_result = llm_results[0]
        paper = replace(
            paper,
            summary_text=primary_llm_result.summary_text,
            summary_basis=primary_llm_result.summary_basis,
            tags=primary_llm_result.tags,
            llm_summary=primary_llm_result.structured,
            analysis_updated_at=now_iso(settings.timezone),
        )
    else:
        summary_text, summary_basis, tags = build_paper_summary(paper, evaluations)
        paper = replace(
            paper,
            summary_text=summary_text,
            summary_basis=summary_basis,
            tags=tags,
            analysis_updated_at=now_iso(settings.timezone),
        )

    source_urls = [item for item in [candidate.primary_url, candidate.pdf_url] if item]
    paper_add_suggestion = _preview_paper_add_suggestion(args, evaluations)
    paths = generate_preview_report(
        settings,
        paper,
        evaluations,
        source_names=["preview"],
        source_urls=source_urls,
        summaries=llm_summaries,
        llm_variants=enabled_variants,
        paper_add_suggestion=paper_add_suggestion,
    )

    relevant_topics = [item.topic_name for item in evaluations if item.classification == "relevant"]
    maybe_topics = [item.topic_name for item in evaluations if item.classification == "maybe"]
    print(f"预分析已生成: {paths['markdown']}")
    print(f"HTML: {paths['html']}")
    print(f"JSON: {paths['json']}")
    print(f"默认总结来源: {paper.summary_basis}")
    print(f"PDF 状态: {paper.pdf_status} / 全文状态: {paper.fulltext_status}")
    print(f"相关领域: {', '.join(relevant_topics) or '无'}")
    if maybe_topics:
        print(f"可能相关领域: {', '.join(maybe_topics)}")
    if paper_add_suggestion.get("command"):
        print("如需入库，可执行:")
        print(paper_add_suggestion["command"])
    else:
        print(paper_add_suggestion.get("note") or "当前没有推荐入库领域，请手动确认 topic 后再执行 paper-add。")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.command == "init":
        config_target = args.config
        if config_target == DEFAULT_RUNTIME_CONFIG:
            config_target = DEFAULT_INIT_CONFIG
        config_path = write_default_config(config_target, force=args.force)
        settings = load_settings(config_path)
        ensure_directory(settings.report_dir)
        ensure_directory(settings.export_dir)
        ensure_directory(settings.database_path.parent)
        ensure_directory(settings.enrichment.pdf_dir)
        ensure_directory(settings.enrichment.text_dir)
        db = Database(settings.database_path, settings.timezone)
        db.initialize()
        db.close()
        print(f"已初始化配置: {config_path}")
        print(f"数据库路径: {settings.database_path}")
        return 0

    if args.command == "compare-report":
        left_settings = load_settings(args.left_config)
        right_settings = load_settings(args.right_config)
        if [topic.id for topic in left_settings.topics] != [topic.id for topic in right_settings.topics]:
            raise ValueError("compare-report 需要两份配置使用相同 topic 列表。")
        if left_settings.database_path != right_settings.database_path:
            raise ValueError("compare-report 需要两份配置指向同一个数据库，以便对比同一批论文。")

        compare_db = Database(left_settings.database_path, left_settings.timezone)
        compare_db.initialize()
        try:
            report_date = args.date or today_string(left_settings.timezone)
            variants = build_runtime_variants(left_settings, [args.right_config])
            paths = generate_comparison_report(
                compare_db,
                left_settings,
                report_date=report_date,
                report_type=args.type,
                variants=variants,
                lookback_days=args.days,
            )
        finally:
            compare_db.close()

        print(f"对比报告已生成: {paths['markdown']}")
        print(f"HTML: {paths['html']}")
        print(f"JSON: {paths['json']}")
        return 0

    if args.command == "paper-preview":
        return _run_paper_preview(args)

    db, pipeline = _open_database(args.config)
    settings = pipeline.settings
    extra_configs = getattr(args, "extra_config", None) or []
    runtime_variants = build_runtime_variants(settings, extra_configs)
    enabled_variants = [variant for variant in runtime_variants if variant.client.enabled]
    digest_variants = enabled_variants[:1] if getattr(args, "digest_primary_only", False) and enabled_variants else enabled_variants
    enrichment_pipeline = EnrichmentPipeline(settings, db, llm_variants=runtime_variants)
    llm_client = enabled_variants[0].client if enabled_variants else LLMClient(settings.llm)
    if getattr(args, "with_llm", False) and not enabled_variants:
        LOGGER.warning(
            "检测到 --with-llm，但当前没有可用的 LLM 变体。请检查 API key 环境变量和额外配置。"
        )
    try:
        if args.command == "paper-add":
            topics = _select_topics(settings, args.topic)
            source_key = args.primary_url or args.pdf_url or args.doi or args.arxiv_id or args.title
            source_paper_id = f"manual-{stable_hash(source_key)}"
            candidate = PaperCandidate(
                source_name="manual",
                source_paper_id=source_paper_id,
                query_text="manual",
                title=args.title,
                abstract=args.abstract or "",
                authors=args.author or [],
                published_at=args.published_at,
                updated_at=args.updated_at,
                primary_url=args.primary_url or args.pdf_url or "",
                pdf_url=args.pdf_url or "",
                doi=args.doi or "",
                arxiv_id=args.arxiv_id or "",
                venue=args.venue or "manual",
                year=args.year,
                categories=args.category or [],
                raw={"manual": True, "added_via": "paper-add"},
            )
            paper_id, created = db.upsert_paper(candidate)
            for topic in topics:
                db.upsert_match(
                    paper_id,
                    TopicEvaluation(
                        topic_id=topic.id,
                        topic_name=topic.display_name,
                        score=max(float(topic.threshold), 100.0),
                        classification=args.classification,
                        matched_keywords=["manual"],
                        reasons=["用户手动加入论文"],
                    ),
                )
            print(f"论文已写入数据库: paper_id={paper_id}, {'新建' if created else '更新'}")
            print(f"所属领域: {', '.join(topic.display_name for topic in topics)}")
            return 0

        if args.command == "paper-set-pdf":
            try:
                paper = db.set_paper_pdf_source(args.paper_id, args.pdf_url)
            except KeyError:
                print(f"未找到论文: paper_id={args.paper_id}")
                return 1
            print(f"已更新 PDF 源: paper_id={paper.id}")
            print(f"标题: {paper.title}")
            print(f"PDF URL: {paper.pdf_url}")
            print("已重置 PDF / 全文缓存状态；不会修改主题归类、评分或已有摘要。")
            print(
                "建议下一步执行: "
                f"python3 main.py --config {shlex.quote(args.config)} enrich --paper-id {paper.id} --with-llm --workers 1 --force"
            )
            return 0

        if args.command == "paper-delete":
            deleted = 0
            missing: list[int] = []
            for paper_id in list(dict.fromkeys(args.paper_id)):
                if db.delete_paper(paper_id):
                    deleted += 1
                else:
                    missing.append(paper_id)
            print(f"删除完成: deleted={deleted}")
            if missing:
                print("未找到的 paper_id:")
                for item in missing:
                    print(f"- {item}")
            return 0

        if args.command == "paper-find":
            if args.no_pdf:
                matches = db.find_papers(
                    title_substring=args.title,
                    url_substring=args.url,
                    doi=args.doi,
                    arxiv_id=args.arxiv_id,
                    topic_ids=args.topic,
                    no_pdf=True,
                    limit=None,
                )
                if not matches:
                    print("没有找到匹配的论文。")
                    return 0
                display_matches = sorted(matches, key=lambda item: item.id)[: max(int(args.limit or len(matches)), 1)]
                print(f"当前库里没有本地 PDF 的论文有 {len(matches)} 篇：")
                source_counter: Counter[str] = Counter()
                for paper in display_matches:
                    print(f"- {paper.id} {paper.title}")
                    if paper.pdf_url:
                        print(f"  当前 pdf_url: {paper.pdf_url}")
                    if paper.primary_url:
                        print(f"  当前 primary_url: {paper.primary_url}")
                    if paper.doi:
                        print(f"  DOI: {paper.doi}")
                    if paper.arxiv_id:
                        print(f"  arXiv: {paper.arxiv_id}")
                    source_mode = ""
                    if isinstance(paper.llm_summary, dict):
                        source_mode = str(paper.llm_summary.get("source_mode") or "").strip()
                    summary_basis = (paper.summary_basis or "metadata-only").strip() or "metadata-only"
                    source_label = f"{summary_basis} / {source_mode or 'unknown'}"
                    source_counter[source_label] += 1
                    if args.show_summary:
                        summary_text = (paper.summary_text or "").strip()
                        print("  当前默认摘要:")
                        if summary_text:
                            for line in summary_text.splitlines():
                                print(f"    {line}")
                        else:
                            print("    （无）")
                if source_counter:
                    print("\n这些论文当前的默认摘要来源：")
                    for label, count in sorted(source_counter.items(), key=lambda item: (-item[1], item[0])):
                        print(f"- {label}: {count}")
                return 0
            matches = db.find_papers(
                title_substring=args.title,
                url_substring=args.url,
                doi=args.doi,
                arxiv_id=args.arxiv_id,
                topic_ids=args.topic,
                no_pdf=False,
                limit=args.limit,
            )
            if not matches:
                print("没有找到匹配的论文。")
                return 0
            print(f"找到 {len(matches)} 篇论文:")
            for paper in matches:
                evaluations = db.fetch_paper_evaluations(paper.id)
                topics_text = ", ".join(
                    f"{item.topic_id}:{item.classification}:{item.score:g}" for item in evaluations[:4]
                ) or "无"
                print(f"- paper_id={paper.id}")
                print(f"  标题: {paper.title}")
                print(f"  领域: {topics_text}")
                print(f"  发表: {paper.published_at or paper.created_at or '未知'}")
                print(f"  PDF: {'有' if paper.pdf_local_path else '无'} / 全文: {paper.fulltext_status or 'empty'}")
                if paper.primary_url:
                    print(f"  链接: {paper.primary_url}")
                if args.show_summary:
                    print(f"  默认摘要来源: {paper.summary_basis or 'metadata-only'}")
                    print("  默认摘要:")
                    summary_text = (paper.summary_text or "").strip()
                    if summary_text:
                        for line in summary_text.splitlines():
                            print(f"    {line}")
                    else:
                        print("    （无）")
            return 0

        if args.command == "fetch":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(
                selected_sources=selected_sources or None,
                start_year=args.start_year,
                recent_limit=args.recent_limit,
                page_size=args.page_size,
                since_last_run=args.since_last_run,
            )
            print(
                f"抓取完成: fetched={stats.fetched}, processed={stats.processed}, "
                f"stored={stats.stored}, matched={stats.matched}"
            )
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "enrich":
            if args.skip_pdf and args.with_llm:
                LOGGER.warning("当前使用了 --skip-pdf，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")
            enrich_started_at = now_iso(settings.timezone) if args.since_last_run else None
            stats = enrichment_pipeline.run(
                limit=args.limit,
                topic_ids=args.topic,
                classifications=args.classification,
                created_after=db.get_checkpoint(ENRICH_CHECKPOINT_KEY) if args.since_last_run else None,
                paper_ids=args.paper_id,
                force=args.force,
                use_llm=args.with_llm or None,
                skip_document_processing=args.skip_pdf,
                workers=args.workers,
                secondary_priority_only=args.secondary_on_priority,
                secondary_top_per_topic=args.secondary_top_per_topic,
                secondary_min_score=args.secondary_min_score,
                retry_from_variant=args.retry_from_variant,
                retry_from_status=args.retry_from_status,
            )
            if args.since_last_run and enrich_started_at is not None and not stats.errors:
                db.set_checkpoint(ENRICH_CHECKPOINT_KEY, enrich_started_at)
            print(
                f"增强完成: enriched={stats.enriched}, downloaded_pdfs={stats.downloaded_pdfs}, "
                f"extracted_texts={stats.extracted_texts}, llm_summaries={stats.llm_summaries}, skipped={stats.skipped}"
            )
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "report":
            report_date = args.date or today_string(settings.timezone)
            paths = generate_report(
                db,
                settings,
                report_date=report_date,
                report_type=args.type,
                lookback_days=args.days,
                llm_client=llm_client,
                llm_variants=enabled_variants,
                topic_digest_variants=digest_variants,
                use_llm_topic_digest=args.with_llm,
            )
            print(f"报告索引已生成: {paths['markdown']}")
            print(f"HTML 索引: {paths['html']}")
            print(f"JSON 索引: {paths['json']}")
            print(f"主题拆分目录: {paths['topic_dir']}")
            print(f"单篇报告目录: {paths['papers_dir']}")
            return 0

        if args.command == "catalog-report":
            paths = generate_catalog_report(
                db,
                settings,
                topic_ids=args.topic,
                classifications=args.classification,
                llm_variants=enabled_variants,
                topic_digest_variants=digest_variants,
                use_llm_topic_digest=args.with_llm,
            )
            print(f"总览索引已生成: {paths['markdown']}")
            print(f"HTML 索引: {paths['html']}")
            print(f"JSON 索引: {paths['json']}")
            print(f"主题拆分目录: {paths['topic_dir']}")
            print(f"单篇报告目录: {paths['papers_dir']}")
            return 0

        if args.command == "paper-report":
            if args.paper_id:
                paper_ids = args.paper_id
            else:
                report_date = args.date or today_string(settings.timezone)
                lookback_days = args.days or settings.report.lookback_days
                start_at, end_at = to_day_bounds(report_date, settings.timezone, lookback_days)
                entries = db.fetch_report_entries(start_at, end_at, settings.report.include_maybe)
                paper_ids = [entry.paper.id for entry in entries]
            unique_paper_ids = list(dict.fromkeys(paper_ids))
            progress_bar = ProgressBar("单篇报告", len(unique_paper_ids) or 1)
            outputs = generate_paper_reports(
                db,
                settings,
                unique_paper_ids,
                llm_variants=enabled_variants,
                progress_bar=progress_bar,
            )
            progress_bar.close(f"单篇报告已生成 {len(outputs)} 篇")
            if not outputs:
                print("没有找到可导出的论文。")
                return 0
            first = next(iter(outputs.values()))
            print(f"单篇报告数量: {len(outputs)}")
            print(f"示例 Markdown: {first['markdown']}")
            print(f"示例 HTML: {first['html']}")
            print(f"示例 JSON: {first['json']}")
            return 0

        if args.command == "run-once":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(
                selected_sources=selected_sources or None,
                start_year=args.start_year,
                recent_limit=args.recent_limit,
                page_size=args.page_size,
                since_last_run=args.since_last_run,
            )
            enrichment_stats = None
            if args.enrich:
                if args.skip_pdf and args.with_llm:
                    LOGGER.warning("当前使用了 --skip-pdf，LLM 将无法读取完整 PDF 文本，只会基于标题/摘要生成总结。")
                enrich_started_at = now_iso(settings.timezone) if args.since_last_run else None
                enrichment_stats = enrichment_pipeline.run(
                    limit=args.enrich_limit,
                    paper_ids=stats.new_paper_ids if args.since_last_run else None,
                    force=False,
                    use_llm=args.with_llm or None,
                    skip_document_processing=args.skip_pdf,
                    workers=args.workers,
                    secondary_priority_only=args.secondary_on_priority,
                    secondary_top_per_topic=args.secondary_top_per_topic,
                    secondary_min_score=args.secondary_min_score,
                    retry_from_variant=args.retry_from_variant,
                    retry_from_status=args.retry_from_status,
                )
                if args.since_last_run and enrich_started_at is not None and not enrichment_stats.errors:
                    db.set_checkpoint(ENRICH_CHECKPOINT_KEY, enrich_started_at)
            report_date = args.date or today_string(settings.timezone)
            paths = generate_report(
                db,
                settings,
                report_date=report_date,
                report_type=args.type,
                lookback_days=args.days,
                llm_client=llm_client,
                llm_variants=enabled_variants,
                topic_digest_variants=digest_variants,
                use_llm_topic_digest=args.with_llm,
            )
            print(
                f"运行完成: fetched={stats.fetched}, processed={stats.processed}, "
                f"stored={stats.stored}, matched={stats.matched}"
            )
            if enrichment_stats is not None:
                print(
                    f"增强结果: enriched={enrichment_stats.enriched}, "
                    f"downloaded_pdfs={enrichment_stats.downloaded_pdfs}, "
                    f"extracted_texts={enrichment_stats.extracted_texts}, "
                    f"llm_summaries={enrichment_stats.llm_summaries}, skipped={enrichment_stats.skipped}"
                )
            print(f"报告路径: {paths['markdown']}")
            print(f"单篇报告目录: {paths['papers_dir']}")
            if stats.errors:
                print("错误:")
                for item in stats.errors:
                    print(f"- {item}")
            if enrichment_stats and enrichment_stats.errors:
                print("增强错误:")
                for item in enrichment_stats.errors:
                    print(f"- {item}")
            return 0

        if args.command == "daemon":
            run_daemon(
                settings,
                db,
                pipeline,
                report_type=args.type,
                loop_limit=args.loops,
                enrich=args.enrich,
                use_llm=args.with_llm or None,
                llm_variants=enabled_variants,
                skip_document_processing=args.skip_pdf,
                workers=args.workers,
                since_last_run=args.since_last_run,
                digest_primary_only=args.digest_primary_only,
                secondary_priority_only=args.secondary_on_priority,
                secondary_top_per_topic=args.secondary_top_per_topic,
                secondary_min_score=args.secondary_min_score,
                retry_from_variant=args.retry_from_variant,
                retry_from_status=args.retry_from_status,
            )
            return 0

        parser.error(f"未知命令: {args.command}")
        return 2
    finally:
        db.close()
