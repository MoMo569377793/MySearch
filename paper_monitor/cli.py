from __future__ import annotations

import argparse
import logging

from paper_monitor.config import load_settings, write_default_config
from paper_monitor.enrichment import EnrichmentPipeline
from paper_monitor.llm import LLMClient
from paper_monitor.pipeline import MonitorPipeline
from paper_monitor.reports import generate_report
from paper_monitor.scheduler import run_daemon
from paper_monitor.storage import Database
from paper_monitor.utils import ensure_directory, today_string


LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地论文监控工具")
    parser.add_argument("--config", default="config/config.json", help="配置文件路径")
    parser.add_argument("--log-level", default="INFO", help="日志级别，默认 INFO")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="创建默认配置并初始化目录")
    init_parser.add_argument("--force", action="store_true", help="覆盖现有配置文件")

    fetch_parser = subparsers.add_parser("fetch", help="仅抓取并写入数据库")
    fetch_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])

    enrich_parser = subparsers.add_parser("enrich", help="下载 PDF、抽取全文并刷新摘要")
    enrich_parser.add_argument("--limit", type=int, help="本轮最多增强多少篇论文")
    enrich_parser.add_argument("--topic", action="append", help="仅增强指定 topic id")
    enrich_parser.add_argument("--classification", action="append", choices=["relevant", "maybe"])
    enrich_parser.add_argument("--force", action="store_true", help="忽略缓存，重新下载或重新抽取")
    enrich_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则强制尝试生成 LLM 摘要")

    report_parser = subparsers.add_parser("report", help="仅生成报告")
    report_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    report_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    report_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    report_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则生成主题级聚合摘要")

    run_once_parser = subparsers.add_parser("run-once", help="抓取后立即生成报告")
    run_once_parser.add_argument("--date", help="报告日期，格式 YYYY-MM-DD，默认今天")
    run_once_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    run_once_parser.add_argument("--days", type=int, help="统计窗口天数，默认使用配置中的 report.lookback_days")
    run_once_parser.add_argument("--source", action="append", choices=["arxiv", "dblp", "google_scholar_alerts"])
    run_once_parser.add_argument("--enrich", action="store_true", help="抓取后执行全文增强")
    run_once_parser.add_argument("--enrich-limit", type=int, help="本轮增强的论文数量上限")
    run_once_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")

    daemon_parser = subparsers.add_parser("daemon", help="持续轮询抓取，并覆盖更新当天报告")
    daemon_parser.add_argument("--type", default="daily", choices=["daily", "weekly"])
    daemon_parser.add_argument("--loops", type=int, help="仅运行指定轮次，便于测试")
    daemon_parser.add_argument("--enrich", action="store_true", help="每轮抓取后执行全文增强")
    daemon_parser.add_argument("--with-llm", action="store_true", help="若已配置 LLM，则启用 LLM 摘要")

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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.log_level)

    if args.command == "init":
        config_path = write_default_config(args.config, force=args.force)
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

    db, pipeline = _open_database(args.config)
    settings = pipeline.settings
    enrichment_pipeline = EnrichmentPipeline(settings, db)
    llm_client = LLMClient(settings.llm)
    if getattr(args, "with_llm", False) and not llm_client.enabled:
        LOGGER.warning(
            "检测到 --with-llm，但当前 LLM 未启用。请检查 llm.enabled、base_url、model 和 API key 环境变量 `%s`。",
            settings.llm.api_key_env,
        )
    try:
        if args.command == "fetch":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(selected_sources=selected_sources or None)
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
            stats = enrichment_pipeline.run(
                limit=args.limit,
                topic_ids=args.topic,
                classifications=args.classification,
                force=args.force,
                use_llm=args.with_llm or None,
            )
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
                use_llm_topic_digest=args.with_llm,
            )
            print(f"报告已生成: {paths['markdown']}")
            print(f"HTML: {paths['html']}")
            print(f"JSON: {paths['json']}")
            return 0

        if args.command == "run-once":
            selected_sources = set(args.source or [])
            stats = pipeline.run_fetch(selected_sources=selected_sources or None)
            enrichment_stats = None
            if args.enrich:
                enrichment_stats = enrichment_pipeline.run(
                    limit=args.enrich_limit,
                    force=False,
                    use_llm=args.with_llm or None,
                )
            report_date = args.date or today_string(settings.timezone)
            paths = generate_report(
                db,
                settings,
                report_date=report_date,
                report_type=args.type,
                lookback_days=args.days,
                llm_client=llm_client,
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
            )
            return 0

        parser.error(f"未知命令: {args.command}")
        return 2
    finally:
        db.close()
