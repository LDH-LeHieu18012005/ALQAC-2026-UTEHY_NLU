"""Chunk denoiser: biến các chunk THÔ (rời rạc, nhiễu tường thuật) đã lấy qua API
(``outputs/agent_v4_results.json``) thành TEXT TỐI ƯU CHO EMBEDDING/RETRIEVE LUẬT.

Vì sao cần bước này (bằng chứng từ literature, không phải suy đoán):
- COLIEE 2024 / "In-Depth Comprehension of Case Relevance" (arXiv 2404.00947): trước
  khi tính relevance phải LOẠI procedural metadata (thời gian, tòa, tên đương sự, địa
  điểm) và bỏ đoạn phân mảnh — "giảm tối đa nhiễu không đóng góp cho relevance".
- AQgR (arXiv 2508.04710) / ViCSR (SIGIR): structured summary tính similarity THEO
  TỪNG FIELD thay vì cả khối → tránh embedding loss.
- SAILER: án gồm Procedure/Fact/Reasoning/Decision/Tail; retrieve luật chỉ cần
  Fact + Reasoning + Claim, bỏ Procedure/Tail (án phí, thủ tục).

Điều luật viết về KHÁI NIỆM ("bồi thường thiệt hại do súc vật gây ra"), không viết về
"chị N", "10kg", "17 giờ ngày 23/6/2017". Nên nhiễu tường thuật kéo lệch embedding.

Module này KHÔNG gọi LLM (chạy offline ngay, tất định — hợp giai đoạn baseline). Nó:
1. Gộp các fragment trùng/gần trùng (chunk API chồng lấn nhiều).
2. Tách câu, STRIP nhiễu thủ tục (tên người, ngày giờ, địa chỉ, khoảng cách/khối
   lượng, số tiền, số hiệu văn bản) — GIỮ năm (luật có phiên bản theo năm).
3. Chỉ giữ câu còn TÍN HIỆU PHÁP LÝ (yêu cầu/phản bác/chứng cứ/nhận định Tòa/diễn biến).
4. Phân câu về field để retrieve đa truy vấn (multi-query) như AQgR.

Đầu ra cắm thẳng vào ``law_substantive.retrieve_law_substantive(query, ...)``.

Nâng cấp tùy chọn (KHÔNG làm ở đây): thêm 1 lượt LLM viết lại thành thuật ngữ luật
(Athena +10% Hit@5, LegalMALR query understanding) — xếp sau bước denoise tất định này.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..config import get
from .artifacts import index_by_case, load_records
from .chunk_summary import (
    _CLAIM_RELIEF_PATTERNS,
    _COURT_SOURCE_PATTERNS,
    _EVIDENCE_PATTERNS,
    _HONORIFIC_NAME_RE,
    _RESPONSE_PATTERNS,
    _SENTENCE_BOUNDARY_RE,
    _content_tokens,
    _normalise_ws,
)

# --- Field cho multi-query (đủ để retrieve; bỏ Procedure/Tail) ------------------
FIELD_TITLES = {
    "su_viec": "Diễn biến / quan hệ tranh chấp",
    "yeu_cau": "Yêu cầu khởi kiện / phản tố",
    "phan_bac": "Trình bày, phản bác của các bên",
    "chung_cu": "Chứng cứ, tài liệu",
    "toa_an": "Nhận định / quyết định của Tòa",
}
FIELD_ORDER = tuple(FIELD_TITLES)

# --- Nhiễu THỦ TỤC cần loại trước khi embed (arXiv 2404.00947) ------------------
# Thứ tự áp dụng có chủ đích: ngày/giờ trước, rồi địa chỉ, đo lường, tiền, số hiệu.
_NOISE_PATTERNS: tuple[re.Pattern, ...] = (
    # ngày tháng đầy đủ: 23/6/2017, 20-02-2024, "ngày 23/6/2017", "ngày 23 tháng 6 năm 2017"
    re.compile(r"\bngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", re.UNICODE),
    re.compile(r"\bngày\s+\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}", re.UNICODE),
    re.compile(r"\b\d{1,2}\s*[/-]\s*\d{1,2}\s*[/-]\s*\d{2,4}\b", re.UNICODE),
    re.compile(r"\bngày\s+\d{1,2}\b(?!\s*tháng)", re.UNICODE),
    # giờ giấc: "17 giờ", "17 giờ 30 phút", "17h", "17h30"
    re.compile(r"\b\d{1,2}\s*giờ(?:\s*\d{1,2}\s*phút)?", re.UNICODE),
    re.compile(r"\b\d{1,2}h\d{0,2}\b", re.UNICODE),
    # địa chỉ / đơn vị hành chính: "phường P, thành phố L, tỉnh Lào Cai", "tổ 27", "thôn ..."
    re.compile(
        r"\b(?:phường|xã|thị\s*trấn|quận|huyện|thành\s*phố|tỉnh|tổ|thôn|ấp|khu\s*phố|"
        r"số\s*nhà|đường)\s+[^,.;]{1,40}",
        re.UNICODE,
    ),
    # đo lường vật lý không liên quan điều luật: "khoảng 2m", "10kg", "hơn 10 kg", "500 m2"
    re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|km|kg|m2|ha|m|g|l)\b", re.UNICODE),
    # số tiền: "6.535.000đ", "120.000.000 đồng" + phần chữ trong ngoặc
    re.compile(r"\([^)]*(?:nghìn|triệu|tỷ|đồng)[^)]*\)", re.UNICODE),
    re.compile(r"\b\d[\d.,]*\s*đ(?:ồng)?\b", re.UNICODE),
    # số hiệu văn bản/án: "số 12/2017/DS-ST"
    re.compile(r"\bsố\s+\d[\w/\-.]*", re.UNICODE),
)

# Cụm chỉ dấu tình tiết thuần tường thuật (nếu câu CHỈ có mấy cụm này -> bỏ)
_NARRATIVE_ONLY_HINTS = (
    "đi trước", "ngoái lại", "quay lại", "hoảng sợ", "kêu lên", "bám vào",
    "chạy xe", "mải tìm", "cách", "đuổi theo", "lao ra",
)

# Từ khóa TÍN HIỆU PHÁP LÝ để GIỮ câu (gộp từ các pattern đã kiểm chứng trong chunk_summary)
_LEGAL_SIGNAL_PATTERNS: tuple[str, ...] = tuple(dict.fromkeys(
    _CLAIM_RELIEF_PATTERNS
    + _RESPONSE_PATTERNS
    + _EVIDENCE_PATTERNS
    + _COURT_SOURCE_PATTERNS
    + (
        "khởi kiện", "phản tố", "yêu cầu độc lập", "kháng cáo", "nghĩa vụ",
        "trách nhiệm", "hợp đồng", "thiệt hại", "vi phạm", "quyền",
        "súc vật", "tài sản", "thừa kế", "ly hôn", "đất", "quyền sử dụng đất",
    )
))

# Từ khóa phân loại field (tái dùng tinh thần _span_section_score của chunk_summary)
_FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "yeu_cau": (
        "khởi kiện", "phản tố", "yêu cầu độc lập", "kháng cáo", "yêu cầu",
        "bồi thường", "thanh toán", "đòi", "buộc",
    ),
    "phan_bac": (
        "không đồng ý", "không chấp nhận", "cho rằng", "trình bày", "thừa nhận",
        "phủ nhận", "xin", "ý kiến của bị đơn", "phản bác",
    ),
    "chung_cu": (
        "người làm chứng", "biên bản", "giám định", "định giá", "thẩm định",
        "chứng cứ", "tài liệu", "lời khai", "giấy chứng nhận", "xác nhận",
    ),
    "toa_an": (
        "tòa án", "hội đồng xét xử", "tuyên xử", "quyết định", "chấp nhận",
        "bác yêu cầu", "án phí", "nhận định", "viện kiểm sát",
    ),
}

_WS_RE = re.compile(r"\s+")
_ORPHAN_PUNCT_RE = re.compile(r"\s+([,.;:])")
_MULTI_PUNCT_RE = re.compile(r"([,.;:])(\s*[,.;:])+")


def _strip_procedural_noise(text: str) -> str:
    """Loại nhiễu thủ tục (tên người, ngày giờ, địa chỉ, đo lường, tiền, số hiệu)."""
    cleaned = _HONORIFIC_NAME_RE.sub(" ", text)  # "chị Lê Thị T" -> " " (giữ role "bị đơn"...)
    for pattern in _NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = _MULTI_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _ORPHAN_PUNCT_RE.sub(r"\1", cleaned)
    return _WS_RE.sub(" ", cleaned).strip(" ,.;:-")


def _has_legal_signal(text: str) -> bool:
    folded = text.casefold()
    return any(kw in folded for kw in _LEGAL_SIGNAL_PATTERNS)


def _is_narrative_only(text: str) -> bool:
    folded = text.casefold()
    hits_narrative = any(hint in folded for hint in _NARRATIVE_ONLY_HINTS)
    return hits_narrative and not _has_legal_signal(folded)


def _classify_field(text: str) -> str:
    folded = text.casefold()
    best_field, best_score = "su_viec", 0
    for field, keywords in _FIELD_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in folded)
        if score > best_score:
            best_field, best_score = field, score
    return best_field


def _merge_chunk_texts(record: dict) -> list[str]:
    """Gộp text các chunk, bỏ chunk rỗng/trùng chunk_id (chunk API chồng lấn nhiều)."""
    raw_chunks = record.get("evidence_details")
    if not isinstance(raw_chunks, list):
        raw_chunks = record.get("chunks") or []
    texts: list[str] = []
    seen_cid: set[str] = set()
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        cid = str(chunk.get("chunk_id") or chunk.get("hash_id") or "").strip()
        text = _normalise_ws(chunk.get("text"))
        if not text or (cid and cid in seen_cid):
            continue
        if cid:
            seen_cid.add(cid)
        texts.append(text)
    return texts


def _sentences(text: str) -> list[str]:
    normalized = _normalise_ws(text)
    if not normalized:
        return []
    return [
        piece for piece in (_normalise_ws(p) for p in _SENTENCE_BOUNDARY_RE.split(normalized))
        if piece
    ]


def denoise_case(record: dict) -> dict:
    """Chunk thô của 1 case -> text đã lọc nhiễu, chia field, sẵn sàng embed/retrieve."""
    case_id = str(record.get("case_id") or "")
    dup_threshold = float(get("chunk_denoise.duplicate_jaccard", 0.80))
    min_chars = int(get("chunk_denoise.min_sentence_chars", 18))

    raw_texts = _merge_chunk_texts(record)
    raw_chars = sum(len(t) for t in raw_texts)

    kept_by_field: dict[str, list[str]] = {field: [] for field in FIELD_ORDER}
    kept_tokensets: list[set[str]] = []
    keywords: set[str] = set()
    n_raw_sentences = 0
    n_dropped_noise = 0
    n_dropped_dup = 0

    for raw_text in raw_texts:
        for sentence in _sentences(raw_text):
            n_raw_sentences += 1
            clean = _strip_procedural_noise(sentence)
            if len(clean) < min_chars or not _has_legal_signal(clean) or _is_narrative_only(clean):
                n_dropped_noise += 1
                continue
            tokens = _content_tokens(clean)
            is_dup = any(
                (len(tokens & prev) / len(tokens | prev) if tokens and prev else 0.0) >= dup_threshold
                for prev in kept_tokensets
            )
            if is_dup:
                n_dropped_dup += 1
                continue
            kept_tokensets.append(tokens)
            field = _classify_field(clean)
            kept_by_field[field].append(clean)
            for kw in _LEGAL_SIGNAL_PATTERNS:
                if kw in clean.casefold():
                    keywords.add(kw)

    fields = {
        field: " ".join(kept_by_field[field])
        for field in FIELD_ORDER if kept_by_field[field]
    }
    n_kept = sum(len(v) for v in kept_by_field.values())
    # Retrieval text: đưa yêu cầu/tranh chấp lên đầu (foreground key elements — COLIEE)
    ordered_parts = [fields[f] for f in FIELD_ORDER if f in fields]
    retrieval_text = " ".join(ordered_parts)
    clean_chars = len(retrieval_text)

    return {
        "case_id": case_id,
        "retrieval_text": retrieval_text,
        "fields": fields,
        "keywords": sorted(keywords),
        "stats": {
            "n_chunks": len(raw_texts),
            "n_raw_sentences": n_raw_sentences,
            "n_kept_sentences": n_kept,
            "n_dropped_noise": n_dropped_noise,
            "n_dropped_duplicate": n_dropped_dup,
            "raw_chars": raw_chars,
            "clean_chars": clean_chars,
            "noise_reduction_pct": round(100.0 * (1 - clean_chars / raw_chars), 1) if raw_chars else 0.0,
        },
    }


def denoise_all(records: dict[str, dict]) -> list[dict]:
    return [denoise_case(record) for record in records.values()]


def _save(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_cases": len(cases),
        "avg_noise_reduction_pct": round(
            sum(c["stats"]["noise_reduction_pct"] for c in cases) / len(cases), 1
        ) if cases else 0.0,
        "cases": cases,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lọc nhiễu thủ tục khỏi chunk thô -> text tối ưu embed/retrieve luật"
    )
    parser.add_argument("--input", default="outputs/agent_v4_results.json")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--case", dest="case_id")
    target.add_argument("--all", action="store_true")
    parser.add_argument("--out", default="outputs/retrieval_chunks.json")
    parser.add_argument("--json", action="store_true", help="In JSON đầy đủ ra stdout")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    records = index_by_case(load_records(args.input))

    if args.all:
        cases = denoise_all(records)
        _save(Path(args.out), cases)
        for case in cases:
            stats = case["stats"]
            print(
                f"{case['case_id']}: {stats['n_kept_sentences']}/{stats['n_raw_sentences']} câu · "
                f"nhiễu -{stats['noise_reduction_pct']}% · {len(case['fields'])} field",
                file=sys.stderr,
            )
        print(f"[saved] {args.out} · {len(cases)} vụ", file=sys.stderr)
        if args.json:
            print(json.dumps(cases, ensure_ascii=False, indent=2))
        return

    if args.case_id not in records:
        parser.error(f"{args.case_id} không có trong {args.input}")
    result = denoise_case(records[args.case_id])
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for field in FIELD_ORDER:
            if field in result["fields"]:
                print(f"\n### {FIELD_TITLES[field]}\n{result['fields'][field]}")
        print("\n[stats]", json.dumps(result["stats"], ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
