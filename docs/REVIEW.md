# Review — 2026-10-09

Status: reviewed source release. Recovered demo filenames identify Team 7. On 2026-10-09 Samer confirmed team permission to publish the code. Specific individual contributions have not yet been provided and are not inferred.

## Evidence

The working copy passed two source/answer regression tests. A real local CPU run indexed 24 records (six recovered records from each source category) with multilingual-e5-base and FAISS, executed two domain queries and one unrelated query, and returned no sources for the unrelated query. The vectors have 768 dimensions. This is a small execution test, not an accuracy evaluation. No API calls were made. All responses remained `verified=false`. Detailed execution metadata is in `validation-summary.json`; source excerpts were retained only in local intermediate files.

The four canonical local JSONL files contain 102,546 records in total: 50,759 Hadith, 33,715 fatwas, 11,867 Quran records and 6,205 tafsir records. Counts are records rather than unique verses; duplicates, passage grouping, authenticity, permissions, and corpus quality remain unverified. The full corpus was counted but not re-embedded or benchmarked. Several alternative tafsir JSONL files and older index archives exist and are not merged into the release.

The recovered README and snapshot disagree with actual code. The actual search uses E5 embeddings, fusion weights 0.45/0.55, and large source-type boosts; the default RAG agent explicitly disables reranking. bge-m3 appears in a local model cache, which does not prove this version uses it.

## Changes made only in this review copy

- Restore index building when artifacts are missing or rebuilding is requested. The recovered indexer refused to build, making a source-only checkout unusable.
- Hash corpus contents to invalidate indexes after same-count changes. The regression check forced a same-count content change and confirmed rebuilding was selected; embeddings were mocked for that second rebuild only.
- Preserve `ayah_start`, `ayah_end`, `hadith_number` and `source_url` in extracted sources and citation deduplication.
- Limit dotenv loading to the repository's `.env` file, avoiding accidental loading of an unrelated parent-directory credential file.
- Replace stale documentation and specify reviewed core dependencies; keep the UI optional.

Original files and Git history were left untouched. The source manifest hashes the recovered files before these changes.

## Still unresolved

- Samer's exact contribution; team code publication permission has been confirmed.
- Dataset redistribution permissions and authoritative origin/version per source category.
- Corpus relevance and citation correctness against an annotated evaluation set.
- Source-type boosts can outweigh similarity; threshold scores are not calibrated.
- The agent's diagnostics path does not apply the same unrelated-question guard as its normal query path; use the normal interactive entry point for review.
- Optional reranking is not connected to the normal agent. Optional hosted generation uses a Responses API route that has not been tested against Groq.
- Streamlit source exists, but no end-to-end UI session was validated in this review.
- The environment was a new venv with system-site-packages from existing local installs; fresh dependency installation was not tested.
- No project reuse license was selected. Cached pickle indexes should only be loaded from trusted local builds; distributed indexes are excluded.
