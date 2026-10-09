from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

from retrieval_engine import (
    setup_logger,
    ensure_directories,
    route_query,
    search as engine_search,
)
from config import WEAK_SCORE_THRESHOLD
from indexer import load_or_build_index
from data_pipeline import load_processed_records
from answer_generator import AnswerGenerator
from query_optimizer import QueryOptimizer


class RAGAgent:
    def __init__(self, model_name: str | None = None, debug: bool = False) -> None:
        self.logger = setup_logger(debug)
        ensure_directories()

        self.logger.info("Initializing RAGAgent...")
        self.records = load_processed_records(self.logger)
        if not self.records:
            raise RuntimeError("No processed records found. Run the data pipeline first.")

        # Build or load index artifacts
        self.index, self.embeddings, self.tokenized_texts, self.bm25 = load_or_build_index(self.records, rebuild=False, logger=self.logger)

        # Keep the submission build deterministic and avoid loading the reranker.
        self.reranker = None

        # LLM-backed components
        self.query_optimizer = QueryOptimizer()
        self.answer_generator = AnswerGenerator()

    def run_query(self, user_query: str, hybrid_top_k: int = 20, final_top_k: int = 5) -> dict:
        intent = route_query(user_query)
        self.logger.info(f"Intent: {intent}")

        optimized_json = self.query_optimizer.optimize_query(user_query)
        try:
            parsed = json.loads(optimized_json)
            optimized_query = parsed.get("optimized_query", "").strip()
        except Exception:
            optimized_query = user_query

        if not optimized_query:
            optimized_query = user_query

        query_to_search = optimized_query or user_query

        domain_markers = ["حكم", "حلال", "حرام", "فتوى", "فقه", "معنى", "تفسير", "آية", "القرآن", "حديث", "رسول", "نبي", "إسبال", "اسبال", "والدين", "صراط", "مستقيم"]
        if intent == "default" and not any(marker in user_query for marker in domain_markers):
            return {
                "query": user_query,
                "intent": intent,
                "answer": "لم أجد مصادر مرتبطة بالسؤال بدرجة كافية في corpus المتاح.",
                "verified": False,
                "sources": [],
            }

        hybrid_results = engine_search(
            query_to_search,
            self.records,
            self.tokenized_texts,
            self.bm25,
            self.index,
            reranker=None,
            top_k=hybrid_top_k,
        )

        final_results = hybrid_results[:final_top_k]
        final_docs = [dict(self.records[idx], score=float(score)) for idx, score in final_results]

        if not final_docs or not final_results or final_results[0][1] < WEAK_SCORE_THRESHOLD:
            return {
                "query": user_query,
                "intent": intent,
                "answer": "لم أجد مصادر مرتبطة بالسؤال بدرجة كافية.",
                "verified": False,
                "sources": [],
            }

        try:
            response = self.answer_generator.generate_answer(user_query, final_docs)
            if response.get("answer"):
                response["intent"] = intent
                response["optimized_query"] = query_to_search
                return response
        except Exception as exc:
            self.logger.warning("Answer generation failed; falling back to a simple grounded summary: %s", exc)

        fallback_answer = "لم أتمكن من توليد صياغة مباشرة من النموذج، لكن الوثائق المتاحة تشير إلى أن الموضوع مرتبط بالموارد المسترجعة."
        if final_docs:
            top_text = final_docs[0].get("clean_text", "") or final_docs[0].get("text", "")
            if top_text:
                fallback_answer = top_text[:500]

        return {
            "query": user_query,
            "intent": intent,
            "answer": fallback_answer,
            "verified": False,
            "sources": [
                {
                    "source_type": doc.get("source_type", ""),
                    "source": doc.get("source", ""),
                    "book": doc.get("book", ""),
                    "scholar": doc.get("scholar", ""),
                    "surah": doc.get("surah", ""),
                    "ayah": doc.get("ayah", ""),
                    "score": doc.get("score", 0.0),
                }
                for doc in final_docs
            ],
        }

    def run_query_diagnostics(self, user_query: str, hybrid_top_k: int = 20, final_top_k: int = 5) -> dict:
        intent = route_query(user_query)
        optimized_json = self.query_optimizer.optimize_query(user_query)
        try:
            parsed = json.loads(optimized_json)
            optimized_query = parsed.get("optimized_query", "").strip()
        except Exception:
            optimized_query = user_query
        if not optimized_query:
            optimized_query = user_query
        query_to_search = optimized_query or user_query

        # Semantic query used for embedding
        semantic_query = "query: " + query_to_search

        # Preferred sources order for printing
        pref_order = []
        mode = intent if intent in ("hukm", "hadith", "tafsir", "quran") else "default"
        if mode == "hukm":
            pref_order = ["fatwa", "hadith", "quran", "tafsir"]
        elif mode == "hadith":
            pref_order = ["hadith", "fatwa", "tafsir", "quran"]
        elif mode == "tafsir":
            pref_order = ["tafsir", "quran", "hadith", "fatwa"]
        elif mode == "quran":
            pref_order = ["quran", "tafsir", "hadith", "fatwa"]
        else:
            pref_order = ["fatwa", "hadith", "tafsir", "quran"]

        # Hybrid retrieval (returns fused scores)
        hybrid_results = engine_search(
            query_to_search,
            self.records,
            self.tokenized_texts,
            self.bm25,
            self.index,
            reranker=None,
            top_k=hybrid_top_k,
        )

        candidate_docs_before = [dict(self.records[idx], score=float(score)) for idx, score in hybrid_results]

        final_results = hybrid_results[:final_top_k]
        reranker_status = "disabled"

        final_docs = [dict(self.records[idx], score=float(score)) for idx, score in final_results]

        # Relevance threshold check
        top_score = final_results[0][1] if final_results else 0.0
        if reranker_status == "enabled":
            # use configured threshold
            if top_score < WEAK_SCORE_THRESHOLD:
                response = {
                    "query": user_query,
                    "answer": "لم أجد مصادر مرتبطة بالسؤال بدرجة كافية.",
                    "verified": False,
                    "sources": [],
                    "reranker": "enabled",
                }
                return {
                    "raw_query": user_query,
                    "intent": intent,
                    "optimized_query": optimized_query,
                    "semantic_query": semantic_query,
                    "preferred_sources": pref_order,
                    "candidate_docs_before": candidate_docs_before,
                    "final_docs": final_docs,
                    "response": response,
                    "reranker": reranker_status,
                }

        # Generate answer from final docs
        response = self.answer_generator.generate_answer(user_query, final_docs)

        # Attach metadata
        return {
            "raw_query": user_query,
            "intent": intent,
            "optimized_query": optimized_query,
            "semantic_query": semantic_query,
            "preferred_sources": pref_order,
            "candidate_docs_before": candidate_docs_before,
            "final_docs": final_docs,
            "response": response,
            "reranker": reranker_status,
        }


def main() -> None:
    agent = RAGAgent()
    print("RAG Agent ready. Enter queries (type 'exit' to quit).")
    while True:
        try:
            q = input("Enter your query: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        try:
            out = agent.run_query(q)
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as exc:
            print(f"Error: {exc}")


if __name__ == "__main__":
    main()
