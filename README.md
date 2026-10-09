# Arabic Islamic RAG — Reviewed Academic Prototype

An Arabic research assistant developed independently by **Samer Weryam**, an AI student at Taibah University. It retrieves Islamic source passages with lexical and semantic search, then returns an excerpt with source metadata. Samer confirmed on 2026-10-09 that he implemented the full original project himself. Legacy filenames containing a team label do not describe implementation authorship.

## Implemented in this recovered version

- Arabic normalization and intent routing.
- BM25 keyword retrieval and FAISS inner-product semantic retrieval.
- `intfloat/multilingual-e5-base` embeddings. This recovered version does **not** use `BAAI/bge-m3` as its embedding model.
- Actual search fusion uses 0.45 BM25 + 0.55 semantic scores, followed by source-type boosts. These boosts can dominate relevance; scores are not probabilities.
- The default agent returns source excerpts, without making API calls. `verified` remains false.
- An Arabic Streamlit interface exists in `app.py`.
- Optional CrossEncoder code references `BAAI/bge-reranker-v2-m3`; the normal RAG agent explicitly passes no reranker. It is not claimed as an active feature.
- Experimental Groq code exists through the OpenAI SDK. Its use of the Responses API has not been validated against Groq; do not present it as a working hosted integration.

## Setup

Python 3.13 was used for local review. Use a virtual environment:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The embedding model downloads on first use unless cached. CPU inference is supported. No corpus, credentials, serialized indexes, or original Git history is distributed with this review package.

## Bring an authorized corpus

Create these UTF-8 JSONL files under `processed/`:

| File | Source type |
|---|---|
| `hadith/hadith_unified.jsonl` | `hadith` |
| `fatwas/fatwas_unified.jsonl` | `fatwa` |
| `quran/quran_unified.jsonl` | `quran` |
| `tafsir/tafsir_unified.jsonl` | `tafsir` |

Each line must be a JSON object containing `clean_text`, `source_type`, and preferably `embedding_text` prefixed with `passage: `. Keep citation metadata such as `source`, `source_url`, `book`, `scholar`, `hadith_number`, `surah`, `ayah_start`, and `ayah_end`. Empty or unrelated corpora will not establish useful religious retrieval.

The recovered raw Hadith README attributes its source to [AhmedBaset/hadith-json](https://github.com/AhmedBaset/hadith-json) and Sunnah.com. Other recovered source permissions and exact download versions have not been established. Obtain authorized copies and retain their attribution; do not assume this package grants corpus redistribution rights.

```bash
python indexer.py --rebuild-index
python run_rag.py
```

The review copy restores index building, which was disabled in the recovered submission snapshot. Cache checks now include a content hash so changed records with the same count do not silently reuse old vectors.

For the interface:

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

## Review and limitations

See [the review](docs/REVIEW.md) for actual validation and unresolved issues. No precision, recall, answer accuracy, or religious-authority claim is made. Metadata citations do not establish source authenticity or answer correctness. Source-boost ranking and the limited domain filter need measured evaluation. The optional generation/verification route is experimental. The local corpus is excluded until its permissions and provenance are clear.

This is an academic research aid, not a source of binding religious rulings. Consult qualified scholars and primary texts. No reuse license has been selected. Code ownership does not establish third-party dataset or video redistribution rights.
