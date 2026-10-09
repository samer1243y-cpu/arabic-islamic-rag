import argparse
import html
import json
import logging
import re
import unicodedata
from hashlib import sha256
from pathlib import Path

from config import LOG_DIR, OUTPUT_PATHS, PROCESSED_DIRS, PROCESSED_SOURCE_DIRS, RAW_SOURCE_DIRS
from tqdm import tqdm


def ensure_directories() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for directory in PROCESSED_SOURCE_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def setup_logger(debug: bool) -> logging.Logger:
    logger = logging.getLogger("DataPipeline")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOG_DIR / "errors.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.addHandler(console_handler)

    return logger


def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")

    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\u0000-\u001F\u007F-\u009F\u200E\u200F\u202A-\u202E]", " ", text)
    text = re.sub(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED\u08D3-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", "", text)
    text = text.replace("ـ", "")
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي").replace("ة", "ه")
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r'[“”«»‘’…·•]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_json_file(path: Path, logger: logging.Logger) -> list[dict]:
    records: list[dict] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"Malformed JSON file {path}: {exc}. Skipping file.")
        return records
    except Exception as exc:
        logger.error(f"Failed to read JSON file {path}: {exc}")
        return records

    meta = {}
    if isinstance(raw, dict):
        if isinstance(raw.get("hadiths"), list):
            records = raw["hadiths"]
            meta = {"chapters": raw.get("chapters", []), "metadata": raw.get("metadata", {})}
        else:
            list_values = [value for value in raw.values() if isinstance(value, list)]
            if len(list_values) == 1 and all(isinstance(item, dict) for item in list_values[0]):
                records = list_values[0]
            elif all(isinstance(value, dict) for value in raw.values()):
                records = [raw]
            else:
                logger.warning(f"Unsupported JSON object shape in {path}. Skipping file.")
                return records
    elif isinstance(raw, list):
        records = raw
    else:
        logger.warning(f"Unsupported JSON content in {path}. Skipping file.")
        return records

    for record in records:
        if isinstance(record, dict):
            record["__source_meta__"] = meta
        else:
            logger.warning(f"Skipping non-object item in {path}")
    return [record for record in records if isinstance(record, dict)]


def load_jsonl_file(path: Path, logger: logging.Logger) -> list[dict]:
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line_no, raw_line in enumerate(input_file, start=1):
                raw_text = raw_line.strip()
                if not raw_text:
                    continue
                try:
                    item = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Invalid JSON line in {path}:{line_no} - {exc}. Skipping line.")
                    continue
                if isinstance(item, list):
                    records.extend(item)
                elif isinstance(item, dict):
                    records.append(item)
                else:
                    logger.warning(f"Unsupported JSONL item type at {path}:{line_no}. Skipping.")
    except Exception as exc:
        logger.error(f"Failed to read JSONL file {path}: {exc}")
    return records


def load_quran_xml_file(path: Path, logger: logging.Logger) -> list[dict]:
    records: list[dict] = []
    try:
        import xml.etree.ElementTree as ET

        tree = ET.parse(path)
        root = tree.getroot()
        # Tailored parsing for common Quran xml layouts (e.g., quran-uthmani.xml)
        # Prefer explicit <sura> elements containing <aya>/<aya> children.
        sura_nodes = root.findall('.//sura') or root.findall('sura') or []

        # If no explicit sura elements, try children of root named like 'suras' then 'sura'
        if not sura_nodes:
            for child in list(root):
                if child.tag.lower().endswith('suras') or child.tag.lower().endswith('sura'):
                    sura_nodes = child.findall('.//sura') or child.findall('sura') or []
                    if sura_nodes:
                        break

        # If still none, fall back to scanning for verse-like elements across the tree
        if not sura_nodes:
            for verse in root.iter():
                t = verse.tag.lower()
                if 'aya' in t or 'verse' in t or 'ayah' in t:
                    text = (verse.text or '').strip()
                    records.append({'surah': '', 'ayah_start': '', 'ayah_end': '', 'text': text})
            return records

        # Parse each sura node
        for sura_idx, sura in enumerate(sura_nodes, start=1):
            # determine sura id/name from attributes or fallback to index
            sura_name = sura.attrib.get('index') or sura.attrib.get('number') or sura.attrib.get('id') or sura.attrib.get('name') or str(sura_idx)
            # find verse children inside sura
            verse_nodes = [c for c in list(sura) if any(x in c.tag.lower() for x in ('aya', 'ayah', 'verse'))]
            # if none explicitly named, accept all children as verses
            if not verse_nodes:
                verse_nodes = list(sura)

            verse_counter = 0
            for v in verse_nodes:
                # skip non-element text nodes
                if not hasattr(v, 'tag'):
                    continue
                verse_counter += 1
                # try multiple ways to get the verse text
                text = ''
                # prefer attribute 'text' if present (common in quran-uthmani.xml)
                if (v.attrib.get('text') if hasattr(v, 'attrib') else None):
                    text = v.attrib.get('text').strip()
                elif (v.text or '').strip():
                    text = v.text.strip()
                else:
                    # look for inner <text> or <w> nodes
                    inner_texts = []
                    for sub in v.iter():
                        if sub is v:
                            continue
                        if sub.text and sub.text.strip():
                            inner_texts.append(sub.text.strip())
                    text = ' '.join(inner_texts).strip()

                records.append({
                    'surah': sura_name,
                    'ayah_start': str(verse_counter),
                    'ayah_end': str(verse_counter),
                    'text': text,
                })
        return records
    except Exception as exc:
        logger.warning(f"Failed to parse XML {path}: {exc}")
        return records


def gather_source_files() -> list[Path]:
    discovered: list[Path] = []
    # Only search raw source directories. Never read processed outputs.
    search_bases = list(RAW_SOURCE_DIRS)
    for base_dir in search_bases:
        if not base_dir.exists():
            continue
        discovered.extend(sorted(base_dir.rglob("*.jsonl")))
        discovered.extend(sorted(base_dir.rglob("*.json")))
        discovered.extend(sorted(base_dir.rglob("*.xml")))
    return discovered


def _debug_inspect_file(path: Path, logger: logging.Logger) -> None:
    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            logger.info(f"[DISCOVER] {path} -> JSONL first-object keys: {list(obj.keys())[:30]}")
                        elif isinstance(obj, list) and obj:
                            logger.info(f"[DISCOVER] {path} -> JSONL first-line is list, first-item keys: {list(obj[0].keys())[:30] if isinstance(obj[0], dict) else 'list'}")
                    except Exception as exc:
                        logger.warning(f"[DISCOVER] {path} JSONL parse error: {exc}")
                    break
        elif path.suffix.lower() == ".json":
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # if it's a dict with list values, try to show the nested list keys
                    if isinstance(raw.get("hadiths"), list):
                        logger.info(f"[DISCOVER] {path} -> JSON top-level hadiths list (count={len(raw.get('hadiths',[]))})")
                    else:
                        logger.info(f"[DISCOVER] {path} -> JSON top-level keys: {list(raw.keys())[:30]}")
                elif isinstance(raw, list) and raw:
                    logger.info(f"[DISCOVER] {path} -> JSON list, first-item keys: {list(raw[0].keys())[:30] if isinstance(raw[0], dict) else 'list'}")
            except Exception as exc:
                logger.warning(f"[DISCOVER] {path} JSON parse error: {exc}")
        elif path.suffix.lower() == ".xml":
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(path)
                root = tree.getroot()
                children = [child.tag for child in list(root)[:10]]
                logger.info(f"[DISCOVER] {path} -> XML root tag: {root.tag}, child tags sample: {children}")
            except Exception as exc:
                logger.warning(f"[DISCOVER] {path} XML parse error: {exc}")
    except Exception as exc:
        logger.warning(f"[DISCOVER] Unexpected inspect error for {path}: {exc}")


def infer_source_type(path: Path, record: dict) -> str:
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    if "db" in parts and ("by_book" in parts or "by_chapter" in parts):
        return "hadith"
    if "db" in parts and "fatwas" in parts:
        return "fatwa"
    if "tafsir" in parts or "tafsir" in name:
        return "tafsir"
    if "quran" in parts:
        return "quran"
    if any(key in record for key in ["question", "answer", "categories"]):
        return "fatwa"
    if any(key in record for key in ["ayah_start", "ayah_end", "surah", "sura"]):
        return "tafsir"
    if any(key in record for key in ["arabic", "idInBook", "chapterId"]):
        return "hadith"
    return "unknown"


def normalize_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value).strip()


def parse_ayah_range(value) -> tuple[str, str]:
    if not value:
        return "", ""
    text = str(value)
    matches = re.findall(r"(\d+)", text)
    if len(matches) == 1:
        return matches[0], matches[0]
    if len(matches) >= 2:
        return matches[0], matches[1]
    return "", ""


def build_embedding_text(source_type: str, metadata: dict, clean_text: str) -> str:
    if source_type == "hadith":
        return (
            f"النوع: حديث\n"
            f"الكتاب: {metadata.get('book', '')}\n"
            f"الفصل: {metadata.get('chapter', '')}\n"
            f"رقم الحديث: {metadata.get('hadith_number', '')}\n"
            f"النص: {clean_text}"
        )
    if source_type == "fatwa":
        return (
            f"النوع: فتوى\n"
            f"العالم: {metadata.get('scholar', '')}\n"
            f"السؤال: {metadata.get('question', '')}\n"
            f"الجواب: {clean_text}"
        )
    if source_type == "tafsir":
        ayah_range = f"{metadata.get('ayah_start','')} - {metadata.get('ayah_end','')}".strip()
        return (
            f"النوع: تفسير\n"
            f"العالم: {metadata.get('scholar', '')}\n"
            f"السورة: {metadata.get('surah', '')}\n"
            f"الآيات: {ayah_range}\n"
            f"النص: {clean_text}"
        )
    if source_type == "quran":
        ayah_range = f"{metadata.get('ayah_start','')} - {metadata.get('ayah_end','')}".strip()
        return (
            f"النوع: قرآن\n"
            f"السورة: {metadata.get('surah', '')}\n"
            f"الآية: {ayah_range}\n"
            f"النص: {clean_text}"
        )
    return clean_text


def build_hadith_record(path: Path, raw: dict) -> dict | None:
    chapters = raw.get("__source_meta__", {}).get("chapters", [])
    text = raw.get("arabic") or raw.get("text") or raw.get("body") or raw.get("content") or ""
    if not text:
        return None

    book = normalize_value(raw.get("book") or raw.get("book_name") or path.stem)
    chapter = normalize_value(raw.get("chapter") or raw.get("chapterName") or raw.get("chapterId"))
    if not chapter and chapters and raw.get("chapterId") is not None:
        chapter_id = str(raw.get("chapterId"))
        for item in chapters:
            if str(item.get("id")) == chapter_id:
                chapter = normalize_value(item.get("arabic") or item.get("english"))
                break
    hadith_number = normalize_value(raw.get("idInBook") or raw.get("id") or raw.get("hadithNumber") or raw.get("hadith_id"))
    clean_text = normalize_arabic(text)
    embedding_text = normalize_arabic(build_embedding_text("hadith", {
        "book": book,
        "chapter": chapter,
        "hadith_number": hadith_number,
    }, clean_text))

    return {
        "source_type": "hadith",
        "book": book,
        "chapter": chapter,
        "hadith_number": hadith_number,
        "text": text,
        "clean_text": clean_text,
        "embedding_text": embedding_text,
    }


def build_fatwa_record(path: Path, raw: dict) -> dict | None:
    # Accept many possible scholar keys
    scholar = normalize_value(
        raw.get("mufti") or raw.get("scholar") or raw.get("author") or raw.get("source") or raw.get("website_name") or raw.get("url") or "Unknown"
    )

    # Extract question from several possible fields
    question = raw.get("question") or raw.get("title") or raw.get("main_title") or raw.get("sub_title") or raw.get("subject") or raw.get("query") or ""
    if isinstance(question, list):
        question = "\n".join([str(q) for q in question])
    question = normalize_value(question)

    # Extract answer; handle list/dict types
    answer_raw = raw.get("answer") or raw.get("response") or raw.get("text") or raw.get("body") or raw.get("content") or raw.get("answer_text") or ""
    if isinstance(answer_raw, list):
        answer = "\n".join([normalize_value(x) for x in answer_raw])
    elif isinstance(answer_raw, dict):
        # try common nested keys
        answer = normalize_value(answer_raw.get("text") or answer_raw.get("body") or json.dumps(answer_raw, ensure_ascii=False))
    else:
        answer = normalize_value(answer_raw)

    if not answer and not question:
        return None

    clean_text = normalize_arabic(f"{question}\n{answer}".strip())
    embedding_text = normalize_arabic(
        build_embedding_text("fatwa", {"scholar": scholar, "question": question}, clean_text)
    )

    return {
        "source_type": "fatwa",
        "scholar": scholar,
        "question": question,
        "answer": answer,
        "clean_text": clean_text,
        "embedding_text": embedding_text,
    }


def build_tafsir_record(path: Path, raw: dict) -> dict | None:
    scholar = normalize_value(raw.get("scholar") or raw.get("author") or raw.get("source") or raw.get("tafsir_name") or "Unknown")

    # Robust surah extraction from multiple possible keys
    surah = normalize_value(
        raw.get("surah")
        or raw.get("sura")
        or raw.get("chapter")
        or raw.get("surah_number")
        or raw.get("sura_no")
        or raw.get("chapterId")
        or raw.get("book")
        or ""
    )

    # Try to extract surah from ayah_range like '2:5-7' if not present
    ayah_range_raw = raw.get("ayah_range") or raw.get("ayah") or raw.get("verse") or raw.get("ayah_range_text") or ""
    ayah_start, ayah_end = parse_ayah_range(ayah_range_raw)
    if not surah and isinstance(ayah_range_raw, str) and ":" in ayah_range_raw:
        parts = ayah_range_raw.split(":", 1)
        surah_candidate = parts[0].strip()
        if surah_candidate.isdigit():
            surah = surah_candidate
    if not ayah_start:
        ayah_start = normalize_value(raw.get("ayah_start") or raw.get("start") or raw.get("ayah_from"))
        ayah_end = normalize_value(raw.get("ayah_end") or raw.get("end") or raw.get("ayah_to") or ayah_start)

    text = raw.get("text") or raw.get("original_text") or raw.get("clean_text") or raw.get("body") or raw.get("content") or raw.get("data") or ""
    # if it's a dict or list, stringify reasonably
    if isinstance(text, list):
        text = "\n".join([normalize_value(x) for x in text])
    elif isinstance(text, dict):
        # some tafsir JSONs have nested structures under 'data'
        if 'text' in text:
            text = normalize_value(text.get('text'))
        else:
            text = normalize_value(json.dumps(text, ensure_ascii=False))
    text = normalize_value(text)
    if not text:
        return None

    clean_text = normalize_arabic(text)
    embedding_text = normalize_arabic(
        build_embedding_text("tafsir", {"scholar": scholar, "surah": surah, "ayah_start": ayah_start, "ayah_end": ayah_end}, clean_text)
    )

    return {
        "source_type": "tafsir",
        "scholar": scholar,
        "surah": surah,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "text": text,
        "clean_text": clean_text,
        "embedding_text": embedding_text,
    }


def build_quran_record(path: Path, raw: dict) -> dict | None:
    surah = normalize_value(raw.get("surah") or raw.get("chapter") or raw.get("sura") or raw.get("book") or "")
    ayah_start, ayah_end = parse_ayah_range(raw.get("ayah_range") or raw.get("ayah") or raw.get("verse") or "")
    if not ayah_start:
        ayah_start = normalize_value(raw.get("ayah_start") or raw.get("start"))
        ayah_end = normalize_value(raw.get("ayah_end") or raw.get("end") or ayah_start)
    text = normalize_value(raw.get("arabic") or raw.get("text") or raw.get("content") or raw.get("body") or "")
    if not text:
        return None

    clean_text = normalize_arabic(text)
    embedding_text = normalize_arabic(build_embedding_text("quran", {
        "surah": surah,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
    }, clean_text))

    return {
        "source_type": "quran",
        "surah": surah,
        "ayah_start": ayah_start,
        "ayah_end": ayah_end,
        "text": text,
        "clean_text": clean_text,
        "embedding_text": embedding_text,
    }


REQUIRED_FIELDS = {
    "hadith": ["source_type", "book", "chapter", "hadith_number", "text", "clean_text", "embedding_text"],
    "fatwa": ["source_type", "scholar", "question", "answer", "embedding_text"],
    "tafsir": ["source_type", "scholar", "surah", "ayah_start", "ayah_end", "text", "embedding_text"],
    "quran": ["source_type", "surah", "ayah_start", "ayah_end", "text", "embedding_text"],
}


def validate_record(record: dict, source_type: str, logger: logging.Logger) -> bool:
    missing = [field for field in REQUIRED_FIELDS[source_type] if not record.get(field) and record.get(field) != 0]
    if missing:
        logger.warning(f"Skipping invalid {source_type} record because missing fields: {missing}")
        return False
    return True


def build_unified_record(path: Path, raw: dict, logger: logging.Logger) -> dict | None:
    source_type = infer_source_type(path, raw)
    if source_type == "unknown":
        logger.warning(f"Unable to infer source type for record from {path}. Skipping.")
        return None

    builders = {
        "hadith": build_hadith_record,
        "fatwa": build_fatwa_record,
        "tafsir": build_tafsir_record,
        "quran": build_quran_record,
    }
    # Special-case expansion: some tafsir files have 'groups' (many tafsir segments per chapter)
    if source_type == 'tafsir' and isinstance(raw.get('groups'), list):
        results: list[dict] = []
        chap = raw.get('chapter') or raw.get('surah')
        for g in raw.get('groups'):
            group_raw = {
                'surah': chap,
                'ayah_start': g.get('start') or g.get('ayah_start') or g.get('from'),
                'ayah_end': g.get('end') or g.get('ayah_end') or g.get('to'),
                'text': g.get('tafseer') or g.get('text') or g.get('content') or g.get('body') or '',
                'scholar': raw.get('author_name') or raw.get('tafsir_name') or raw.get('source'),
            }
            rec = build_tafsir_record(path, group_raw)
            if rec and validate_record(rec, 'tafsir', logger):
                results.append(rec)
        if not results:
            return None
        return results

    record = builders[source_type](path, raw)
    if record is None:
        return None

    if not validate_record(record, source_type, logger):
        return None
    return record


def record_signature(record: dict) -> str:
    payload = f"{record.get('source_type')}|{record.get('clean_text','')}|{record.get('embedding_text','')}"
    return sha256(payload.encode("utf-8", errors="ignore")).hexdigest()


def process_sources(logger: logging.Logger) -> dict[str, list[dict]]:
    records_by_type: dict[str, list[dict]] = {"hadith": [], "fatwa": [], "tafsir": [], "quran": []}
    seen_signatures: set[str] = set()
    processed_files = gather_source_files()

    if not processed_files:
        logger.warning("No raw JSON/JSONL files were discovered in source directories.")

    # Debug: list discovered files grouped by likely category
    files_by_cat = {"fatwas": [], "tafsir": [], "quran": [], "hadith": [], "other": []}
    for p in processed_files:
        parts = [part.lower() for part in p.parts]
        if any("fatwa" in part for part in parts):
            files_by_cat["fatwas"].append(p)
        elif any("tafsir" in part for part in parts):
            files_by_cat["tafsir"].append(p)
        elif any("quran" in part for part in parts) or p.suffix.lower() == ".xml":
            files_by_cat["quran"].append(p)
        elif any("by_book" in part or "by_chapter" in part or "hadith" in part for part in parts):
            files_by_cat["hadith"].append(p)
        else:
            files_by_cat["other"].append(p)

    for cat, lst in files_by_cat.items():
        if lst:
            logger.info(f"Discovered {len(lst)} files for {cat}, sample: {lst[:10]}")
            # print first-object keys for first few files to help debugging
            for sample in lst[:5]:
                _debug_inspect_file(sample, logger)

    for source_file in tqdm(processed_files, desc="Scanning sources", unit="file"):
        raw_records = []
        if source_file.suffix.lower() == ".jsonl":
            raw_records = load_jsonl_file(source_file, logger)
        elif source_file.suffix.lower() == ".json":
            raw_records = load_json_file(source_file, logger)
        elif source_file.suffix.lower() == ".xml":
            # special-case: Quran XML files
            raw_records = load_quran_xml_file(source_file, logger)

        if not raw_records:
            # show first-line / keys to help debug empty parsing
            _debug_inspect_file(source_file, logger)

        for raw in raw_records:
            if not isinstance(raw, dict):
                continue
            record = build_unified_record(source_file, raw, logger)
            if record is None:
                continue
            # builder may return a single record or a list of records (e.g., tafsir groups)
            record_list = record if isinstance(record, list) else [record]
            for rec in record_list:
                signature = record_signature(rec)
                if signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                records_by_type[rec["source_type"]].append(rec)

    return records_by_type


def write_unified_outputs(records_by_type: dict[str, list[dict]], logger: logging.Logger) -> None:
    for source_type, records in records_by_type.items():
        # OUTPUT_PATHS keys may be plural (e.g., 'fatwas') so accept both forms
        out_path = OUTPUT_PATHS.get(source_type) or OUTPUT_PATHS.get(source_type + 's')
        if not out_path:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(records)} {source_type} records to {out_path}")
        if not records:
            logger.warning(f"Processed output for {source_type} is empty: {out_path}")


def load_processed_records(logger: logging.Logger) -> list[dict]:
    results: list[dict] = []
    for output_path in OUTPUT_PATHS.values():
        if not output_path.exists():
            continue
        if output_path.suffix.lower() == ".jsonl":
            items = load_jsonl_file(output_path, logger)
        else:
            items = load_json_file(output_path, logger)
        for item in items:
            if isinstance(item, dict):
                results.append(item)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified Arabic Islamic JSONL from local sources.")
    parser.add_argument("--debug", action="store_true", help="Show debug logs")
    args = parser.parse_args()

    logger = setup_logger(args.debug)
    ensure_directories()

    logger.info("Starting data pipeline")
    records_by_type = process_sources(logger)
    write_unified_outputs(records_by_type, logger)
    logger.info("Data pipeline finished")


if __name__ == "__main__":
    main()
