"""Grounded Islamic Answer Generator.

This module provides an AnswerGenerator class that generates answers
strictly from retrieved evidence blocks returned by the retrieval engine.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from config import ENABLE_LLM_VERIFICATION, SUBMISSION_MODE

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).with_name(".env"))


class AnswerGenerator:
    GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or self.GROQ_DEFAULT_BASE_URL
        self.logger = logging.getLogger("AnswerGenerator")
        self.client = None
        try:
            self.client = self._configure_client()
        except Exception as exc:
            self.logger.warning("Groq answer generation unavailable: %s", exc)

    def _configure_client(self) -> Any:
        if load_dotenv is None:
            raise ImportError(
                "The python-dotenv package is required for AnswerGenerator. "
                "Install it with `python -m pip install python-dotenv`."
            )

        if openai is None:
            raise ImportError(
                "The openai package is required for AnswerGenerator. "
                "Install it with `python -m pip install openai`."
            )

        if not self.api_key:
            raise ValueError(
                "No GROQ_API_KEY provided. Set GROQ_API_KEY in your .env file or pass api_key explicitly."
            )

        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def build_context(self, retrieved_docs: list[dict]) -> str:
        if not retrieved_docs:
            return ""

        blocks: list[str] = []
        for index, record in enumerate(retrieved_docs, start=1):
            lines = [f"[DOCUMENT {index}]"]

            source = record.get("source") or record.get("book") or record.get("scholar")
            if source:
                lines.append(f"Source: {source}")

            source_type = record.get("source_type")
            if source_type:
                lines.append(f"Source Type: {source_type}")

            if record.get("book"):
                lines.append(f"Book: {record.get('book')}")
            if record.get("scholar"):
                lines.append(f"Scholar: {record.get('scholar')}")
            if record.get("surah"):
                lines.append(f"Surah: {record.get('surah')}")
            if record.get("ayah"):
                lines.append(f"Ayah: {record.get('ayah')}")
            if record.get("hadith_number"):
                lines.append(f"Hadith Number: {record.get('hadith_number')}")
            if record.get("score") is not None:
                lines.append(f"Score: {record.get('score')}")

            text = record.get("text") or record.get("clean_text") or ""
            lines.append("")
            lines.append("TEXT:")
            lines.append(text.strip())
            blocks.append("\n".join(lines))

        return "\n\n---------------------------------\n\n".join(blocks)

    def build_prompt(self, query: str, context: str) -> str:
        system_instructions = (
            "أنت مساعد بحث إسلامي.\n"
            "أنت غير مسموح لك باختراع أي دليل أو معلومات خارجة عن المصادر المقدمة.\n"
            "يجب أن تجيب فقط من الوثائق المتاحة في السياق.\n"
            "إذا كان الجواب غير موجود صراحة في المصادر، فاكتب فقط:\n"
            "لم أجد جواباً صريحاً في المصادر المتاحة.\n"
            "لا تقوم باختراع فتاوى أو أحاديث أو تفاسير.\n"
            "لا تستشهد بمصدر غير موجود في السياق المقدم.\n"
            "بعد كل عبارة واقعية، ضف مصدرها بين قوسين مربعين مثل: [المصدر: ...].\n"
            "أجب باللغة العربية."
        )

        prompt = (
            f"{system_instructions}\n\n"
            "استخدم فقط الوثائق التالية كدليل:\n"
            "<BEGIN CONTEXT>\n"
            f"{context}\n"
            "<END CONTEXT>\n\n"
            f"السؤال: {query}\n"
        )
        return prompt

    def extract_sources(self, retrieved_docs: list[dict]) -> list[dict]:
        unique = {}
        for record in retrieved_docs:
            key = (
                record.get("source_type", ""),
                record.get("source", ""),
                record.get("book", ""),
                record.get("scholar", ""),
                record.get("surah", ""),
                record.get("ayah", ""),
                record.get("ayah_start", ""),
                record.get("ayah_end", ""),
                record.get("hadith_number", ""),
                record.get("source_url", ""),
            )
            existing = unique.get(key)
            score = record.get("score") if record.get("score") is not None else 0.0
            if existing is None or score > existing.get("score", 0.0):
                unique[key] = {
                    "source_type": record.get("source_type", ""),
                    "source": record.get("source", ""),
                    "book": record.get("book", ""),
                    "scholar": record.get("scholar", ""),
                    "surah": record.get("surah", ""),
                    "ayah": record.get("ayah", ""),
                    "ayah_start": record.get("ayah_start", ""),
                    "ayah_end": record.get("ayah_end", ""),
                    "hadith_number": record.get("hadith_number", ""),
                    "source_url": record.get("source_url", ""),
                    "score": score,
                }

        return list(unique.values())

    def _call_model(self, prompt: str) -> str:
        retry_count = 0
        max_retries = 3
        backoff = 2
        while True:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": prompt},
                    ],
                    temperature=0.0,
                    max_output_tokens=400,
                )
            except Exception as exc:
                retry_count += 1
                message = str(exc).lower()
                if retry_count <= max_retries and ("rate limit" in message or "429" in message):
                    sleep_seconds = backoff ** retry_count
                    self.logger.warning(
                        "Rate limit detected, retrying in %s seconds (%s/%s).",
                        sleep_seconds,
                        retry_count,
                        max_retries,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise

            if hasattr(response, "output_text") and response.output_text is not None:
                return response.output_text.strip()

        if hasattr(response, "output"):
            output = response.output
            if output and isinstance(output, list):
                first = output[0]
                if isinstance(first, dict):
                    return first.get("content", "").strip()

        raise RuntimeError("LLM did not return any text output.")

    def _verify_answer(self, answer: str, context: str) -> str:
        verification_prompt = (
            "أجب بكلمة واحدة فقط: SUPPORTED أو UNSUPPORTED.\n"
            "هل يمكن دعم كل عبارة في الإجابة التالية مباشرةً من السياق المقدم؟\n\n"
            f"الإجابة: {answer}\n\n"
            f"السياق: {context}\n"
        )
        retry_count = 0
        max_retries = 3
        backoff = 2
        while True:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[
                        {"role": "system", "content": "أنت مدقق دقة. أجب بكلمة واحدة فقط: SUPPORTED أو UNSUPPORTED."},
                        {"role": "user", "content": verification_prompt},
                    ],
                    temperature=0.0,
                    max_output_tokens=20,
                )
            except Exception as exc:
                retry_count += 1
                message = str(exc).lower()
                if retry_count <= max_retries and ("rate limit" in message or "429" in message):
                    sleep_seconds = backoff ** retry_count
                    self.logger.warning(
                        "Rate limit on verification, retrying in %s seconds (%s/%s).",
                        sleep_seconds,
                        retry_count,
                        max_retries,
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise

        if hasattr(response, "output_text") and response.output_text is not None:
            content = response.output_text.strip().upper()
        else:
            content = ""
            if hasattr(response, "output"):
                output = response.output
                if output and isinstance(output, list):
                    first = output[0]
                    if isinstance(first, dict):
                        content = first.get("content", "").strip().upper()

        if "SUPPORTED" in content and "UNSUPPORTED" not in content:
            return "SUPPORTED"
        if "UNSUPPORTED" in content:
            return "UNSUPPORTED"
        return "UNSUPPORTED"

    def generate_answer(self, query: str, retrieved_docs: list[dict]) -> dict:
        context = self.build_context(retrieved_docs)
        prompt = self.build_prompt(query, context)
        sources = self.extract_sources(retrieved_docs)

        if not self.client or SUBMISSION_MODE or not ENABLE_LLM_VERIFICATION:
            answer_text = self._grounded_fallback(query, retrieved_docs)
            return {
                "query": query,
                "answer": answer_text,
                "verified": False,
                "sources": sources,
            }

        try:
            generated_answer = self._call_model(prompt)
        except Exception as exc:
            message = str(exc).lower()
            if "429" in message or "rate limit" in message:
                return {
                    "query": query,
                    "answer": "تعذر توليد الصياغة الآن بسبب حد الخدمة، وهذه هي المصادر المسترجعة.",
                    "verified": False,
                    "sources": sources,
                }
            self.logger.error("Failed to generate answer: %s", exc)
            return {
                "query": query,
                "answer": self._grounded_fallback(query, retrieved_docs),
                "verified": False,
                "sources": sources,
            }

        verification = self._verify_answer(generated_answer, context)
        if verification != "SUPPORTED":
            return {
                "query": query,
                "answer": self._grounded_fallback(query, retrieved_docs),
                "verified": False,
                "sources": sources,
            }

        answer_text = generated_answer.strip()
        if not answer_text:
            answer_text = self._grounded_fallback(query, retrieved_docs)
        else:
            answer_text = f"{answer_text}\n\nتنبيه: هذا مساعد بحثي وليس فتوى ملزمة."

        return {
            "query": query,
            "answer": answer_text,
            "verified": True,
            "sources": sources,
        }

    def _grounded_fallback(self, query: str, retrieved_docs: list[dict]) -> str:
        if not retrieved_docs:
            return "لم أجد مصادر مرتبطة بالسؤال بدرجة كافية."

        first_doc = retrieved_docs[0]
        text = first_doc.get("clean_text") or first_doc.get("text") or ""
        if not text:
            return "لم أجد مصادر مرتبطة بالسؤال بدرجة كافية."

        source_label = first_doc.get("source") or first_doc.get("book") or first_doc.get("scholar") or first_doc.get("source_type") or "مصدر غير محدد"
        excerpt = " ".join(text.split())
        if len(excerpt) > 500:
            excerpt = excerpt[:500] + "..."
        return f"استندت الإجابة إلى الوثائق المتاحة [المصدر: {source_label}]: {excerpt}\n\nتنبيه: هذا مساعد بحثي وليس فتوى ملزمة."


if __name__ == "__main__":
    # Example usage (integration may require an application wrapper since
    # retrieval_engine.py currently exposes search functions rather than a
    # RetrievalEngine class).
    #
    # from retrieval_engine import RetrievalEngine
    # engine = RetrievalEngine()
    # results = engine.search("ما حكم الإسبال", top_k=10)
    # generator = AnswerGenerator()
    # response = generator.generate_answer("ما حكم الإسبال", results)
    # print(response["answer"])
    print("AnswerGenerator module loaded.")
