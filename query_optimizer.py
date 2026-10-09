from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from config import ENABLE_LLM_QUERY_OPTIMIZATION, SUBMISSION_MODE

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


class QueryOptimizer:
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
        self.logger = logging.getLogger("QueryOptimizer")
        self.client = None
        self.use_llm = (ENABLE_LLM_QUERY_OPTIMIZATION or SUBMISSION_MODE is False) and bool(self.api_key)
        if self.use_llm:
            try:
                self.client = self._configure_client()
            except Exception as exc:
                self.logger.warning("Query optimization LLM unavailable: %s", exc)
                self.client = None

    def _configure_client(self) -> Any:
        if load_dotenv is None:
            raise ImportError(
                "The python-dotenv package is required for QueryOptimizer. "
                "Install it with `python -m pip install python-dotenv`."
            )

        if openai is None:
            raise ImportError(
                "The openai package is required for QueryOptimizer. "
                "Install it with `python -m pip install openai`."
            )

        if not self.api_key:
            raise ValueError(
                "No GROQ_API_KEY provided. Set GROQ_API_KEY in your .env file or pass api_key explicitly."
            )

        return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_model(self, prompt: str, max_tokens: int = 200) -> str:
        retry_count = 0
        max_retries = 3
        backoff = 2
        while True:
            try:
                response = self.client.responses.create(
                    model=self.model_name,
                    input=[{"role": "system", "content": prompt}],
                    temperature=0.0,
                    max_output_tokens=max_tokens,
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

    def optimize_query(self, user_query: str) -> str:
        """
        Transform a user's natural language question into a search-oriented query.

        Returns a raw JSON string containing the single key `optimized_query`.
        The function MUST return only raw JSON (no surrounding markdown or extra text).
        """
        if not isinstance(user_query, str) or not user_query.strip():
            return json.dumps({"optimized_query": ""}, ensure_ascii=False)

        prompt = (
            "أنت خبير NLP إسلامي متخصص في تحويل الأسئلة الطبيعية إلى استعلام بحث مهيأ لمحركات البحث و استرجاع الوثائق.\n"
            "المطلوب: حافظ على المعنى الأصلي، أزل الحشو والمقدمات المحادثية، وسّع المصطلحات الإسلامية المهمة، وأضف مصطلحات مرتبطة عندما يُناسب ذلك.\n"
            "أعدّ استعلاماً مختصراً ومركزاً من كلمات ومصطلحات مفصولة بمسافات لزيادة تغطية البحث.\n"
            "أمثلة: \n"
            "Input: ما حكم القروض؟\n"
            "Output: {\"optimized_query\":\"حكم القروض البنكية الربا القرض الحسن التمويل الإسلامي الاقتراض\"}\n"
            "Input: ما معنى الصراط المستقيم؟\n"
            "Output: {\"optimized_query\":\"تفسير معنى الصراط المستقيم في القرآن تفسير ابن كثير\"}\n"
            "Input: هل الإسبال حرام؟\n"
            "Output: {\"optimized_query\":\"حكم الإسبال إرخاء الثوب تحت الكعبين أحاديث الإسبال فتاوى ابن باز ابن عثيمين\"}\n\n"
            "تنبيهات مهمة:\n"
            "- أعد فقط نص JSON صالح يحتوي على المفتاح \"optimized_query\" وقيمة نصية واحدة.\n"
            "- لا تضف أي شروحات أو علامات ترقيم خارج JSON، ولا تحيط النتيجة بأي كود أو علامات.\n"
            f"User Input: {user_query}\n"
            "Output:"
        )

        if not self.client or SUBMISSION_MODE or not ENABLE_LLM_QUERY_OPTIMIZATION:
            return json.dumps({"optimized_query": self._fallback_optimize_query(user_query)}, ensure_ascii=False)

        # Ask the model
        raw = self._call_model(prompt, max_tokens=200)

        # The model is instructed to return pure JSON. Try to sanitize/trim anything
        text = raw.strip()

        # Attempt to locate the first JSON object in the output.
        # If the model returned extra text, extract JSON substring.
        try:
            # Fast path: if it starts with { parse directly
            if text.startswith("{"):
                # Ensure valid JSON
                parsed = json.loads(text)
                return json.dumps(parsed, ensure_ascii=False)

            # Otherwise, try to find a JSON object inside the text
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = text[start : end + 1]
                parsed = json.loads(candidate)
                return json.dumps(parsed, ensure_ascii=False)
        except Exception:
            # Fall back to best-effort: wrap returned string as optimized_query
            pass

        return json.dumps({"optimized_query": text}, ensure_ascii=False)

    def _fallback_optimize_query(self, user_query: str) -> str:
        text = user_query.strip()
        if not text:
            return ""

        tokens = []
        for part in text.replace("؟", "").replace("?", "").split():
            cleaned = part.strip(" ،؛:().")
            if cleaned:
                tokens.append(cleaned)

        if not tokens:
            return text

        lowered = " ".join(tokens).lower()
        if "إسبال" in lowered or "اسبال" in lowered:
            return "حكم الإسبال إرخاء الثوب أسفل الكعبين"
        if "صراط" in lowered or "مستقيم" in lowered:
            return "تفسير معنى الصراط المستقيم"
        if "والدين" in lowered or "والد" in lowered or "بر الوالدين" in lowered:
            return "حديث بر الوالدين والإحسان إليهما"
        if "حديث" in lowered or "رسول" in lowered or "نبي" in lowered:
            return f"حديث {' '.join(tokens[1:]) if len(tokens) > 1 else ' '.join(tokens)}"

        expansions = []
        if any(word in lowered for word in ["قرض", "اقتراض", "تمويل"]):
            expansions.extend(["قرض", "اقتراض", "تمويل", "ربا"])
        if any(word in lowered for word in ["صراط", "مستقيم"]):
            expansions.extend(["صراط", "مستقيم", "تفسير", "القرآن"])
        if any(word in lowered for word in ["إسبال", "ثوب", "كعب"]):
            expansions.extend(["إسبال", "ثوب", "كعب", "أحاديث"])
        if any(word in lowered for word in ["حكم", "حرام", "جائز"]):
            expansions.extend(["حكم", "فقه", "فتوى"])

        merged = list(dict.fromkeys(tokens + expansions))
        return " ".join(merged)


if __name__ == "__main__":
    print("QueryOptimizer loaded.")
