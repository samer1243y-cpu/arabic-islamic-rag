from __future__ import annotations

import os
import sys
os.environ["PYTHONUTF8"] = "1"
try:
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # reconfigure may not be available in some environments; ignore safely
    pass

import argparse
import json
from pathlib import Path
from typing import Any

from rag_agent import RAGAgent

VALIDATION_QUERIES = [
    "ما معنى الصراط المستقيم؟",
    "ما حكم الإسبال؟",
    "من هو رئيس الولايات المتحدة؟",
]
LOG_PATH = Path("logs") / "end_to_end_validation.json"


def print_query_summary(entry: dict) -> None:
    print("\n" + "=" * 80)
    print(f"السؤال: {entry['raw_query']}")
    print(f"نوع السؤال: {entry['intent']}")
    print(f"الاستعلام المحسن: {entry['optimized_query']}")
    print(f"المصادر النهائية: {', '.join(entry['final_source_types'])}")
    print(f"الإجابة النهائية: {entry['final_answer']}")
    print(f"تم التحقق: {entry['verification_status']}")
    print("المراجع:")
    for src in entry.get("citations", []):
        print(f" - {src}")
    print("=" * 80)


def save_demo_answer(answer_text: str) -> None:
    demo_path = Path("demo_last_answer.txt")
    demo_path.write_text(answer_text, encoding="utf-8")


def print_concise_result(query: str, result: dict, debug: bool = False) -> None:
    answer_text = result.get("answer") or ""
    sources = result.get("sources", [])
    alert_text = "تنبيه: هذا مساعد بحثي وليس فتوى ملزمة."
    print("\nالسؤال:")
    print(query)
    print("\nنوع السؤال:")
    print(result.get("intent") or "unknown")
    print("\nالإجابة النهائية:")
    print(answer_text)
    print("\nالمصادر:")
    if sources:
        for source in sources[:5]:
            print(f"- {source.get('source_type', '')}: {source.get('book') or source.get('scholar') or source.get('source') or source.get('surah') or source.get('ayah') or ''}")
    else:
        print("- لا توجد مصادر.")
    print("\nالتنبيه:")
    print(alert_text)
    if debug:
        print("\nتفاصيل تقنية:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    save_demo_answer(f"{query}\n\n{answer_text}\n\n{alert_text}")


def interactive_loop(agent: RAGAgent, debug: bool = False) -> None:
    print("Islamic RAG pipeline ready. Enter query or type 'exit'.")
    while True:
        try:
            query = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        result = agent.run_query(query, final_top_k=5)
        print_concise_result(query, result, debug=debug)


def validate(agent: RAGAgent) -> list[dict]:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for query in VALIDATION_QUERIES:
        diagnostic = agent.run_query_diagnostics(query)
        response = diagnostic.get("response", {})
        entry = {
            "raw_query": query,
            "intent": diagnostic["intent"],
            "optimized_query": diagnostic["optimized_query"],
            "candidate_source_types_before": [
                f"{item['source_type']} ({item.get('book') or item.get('scholar') or item.get('surah') or item.get('hadith_number') or 'unknown'})"
                for item in diagnostic["candidate_docs_before"]
            ],
            "final_source_types": [
                f"{item['source_type']} ({item.get('book') or item.get('scholar') or item.get('surah') or item.get('hadith_number') or 'unknown'})"
                for item in diagnostic["final_docs"]
            ],
            "final_answer": response.get("answer", ""),
            "verification_status": response.get("verified", False),
            "citations": [
                f"{src.get('source_type','')} - {src.get('book') or src.get('scholar') or src.get('source') or src.get('surah') or src.get('ayah') or src.get('hadith_number')}"
                for src in response.get("sources", [])
            ],
            "raw_response": response,
        }
        print_query_summary(entry)
        results.append(entry)
    LOG_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved validation diagnostics to {LOG_PATH}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Islamic RAG pipeline.")
    parser.add_argument("--validate", action="store_true", help="Run end-to-end validation and save logs/end_to_end_validation.json")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive mode")
    args = parser.parse_args()

    agent = RAGAgent(debug=args.debug)

    if args.validate:
        validate(agent)

    if not args.no_interactive and not args.validate:
        interactive_loop(agent, debug=args.debug)


if __name__ == "__main__":
    main()
