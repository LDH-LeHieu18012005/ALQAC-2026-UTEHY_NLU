"""Legal-query rewrite (Athena arXiv 2410.11195 / LegalMALR): biến mô tả vụ án phân
mảnh -> TRUY VẤN PHÁP LÝ bằng THUẬT NGỮ LUẬT, để retrieve ĐIỀU LUẬT NỘI DUNG.

Vì sao hướng này (đã kiểm chứng thực nghiệm 2026-07-22, hơn cả chunk_summary):
- chunk_summary (2 lượt, validate ngặt) hợp JUDGE nhưng MẤT khái niệm pháp lý khi
  bóc tách vụ phân mảnh (fact "chị N thấy chó" sống, claim "bồi thường" chết).
- Retrieve luật cần KHÁI NIỆM ("trách nhiệm bồi thường thiệt hại do súc vật gây ra"),
  không cần tường thuật. Athena: rewrite case -> mô tả pháp lý -> +10% Hit@5.

1 lượt LLM/vụ (rẻ hơn chunk_summary 2 lượt). Đầu vào = case_query (sạch, có sẵn) +
text đã lọc nhiễu regex (chunk_denoise). KHÔNG phán thắng thua, KHÔNG bịa số điều.
Đầu ra `outputs/retrieval_chunks_llm.json` (cùng schema chunk_denoise -> eval_recall đọc thẳng).

Chạy: py -X utf8 -m alqac2026.retrieval.legal_query --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..config import get
from ..data_io import raw_case
from .retrieval_from_summary import _clean_case_query

# Tăng khi _SYSTEM/_PROMPT đổi nội dung sinh ra kết quả khác — resume() lọc theo version
# này để KHÔNG âm thầm giữ lại kết quả cũ (case_id đã có trong file) khi prompt đã đổi.
PROMPT_VERSION = "v2_broad_domain"

_SYSTEM = (
    "Bạn là chuyên gia pháp luật Việt Nam, xử lý tranh chấp DÂN SỰ SƠ THẨM theo nghĩa RỘNG "
    "(gồm cả hợp đồng dân sự có yếu tố kinh doanh/thương mại, tín dụng ngân hàng, doanh "
    "nghiệp, đất đai, công chứng — không chỉ dân sự thuần túy). Nhiệm vụ: từ mô tả vụ án, "
    "viết TRUY VẤN PHÁP LÝ để TÌM ĐIỀU LUẬT NỘI DUNG liên quan, đúng đúng LĨNH VỰC PHÁP LUẬT "
    "cụ thể của vụ việc (không mặc định là Bộ luật Dân sự nếu bản chất là quan hệ khác — vd. "
    "vay tín dụng dùng Luật Các tổ chức tín dụng, mua bán hàng hoá dùng Luật Thương mại, "
    "tranh chấp nội bộ công ty dùng Luật Doanh nghiệp). Trả lời hoàn toàn bằng tiếng Việt."
)

_PROMPT = """Cho TÓM TẮT và TÌNH TIẾT (rời rạc, có nhiễu) của một vụ án dân sự sơ thẩm (hiểu \
theo nghĩa rộng, có thể là hợp đồng dân sự, thương mại, tín dụng, doanh nghiệp, đất đai, thừa \
kế, hôn nhân gia đình, công chứng...). Viết một đoạn TRUY VẤN PHÁP LÝ ngắn (3-6 câu) để tìm các \
ĐIỀU LUẬT NỘI DUNG áp dụng.
YÊU CẦU:
- Trước tiên XÁC ĐỊNH ĐÚNG LĨNH VỰC/CHẾ ĐỊNH pháp luật cụ thể của quan hệ tranh chấp (không suy
  diễn mặc định là Bộ luật Dân sự nếu tình tiết cho thấy đây là quan hệ chuyên ngành khác).
- Dùng THUẬT NGỮ PHÁP LÝ (loại quan hệ tranh chấp, chế định, nghĩa vụ, trách nhiệm bồi thường...).
- BỎ tên người, ngày giờ, địa chỉ, số tiền cụ thể.
- Nêu rõ: loại tranh chấp, yêu cầu pháp lý của các bên, chế định/căn cứ pháp lý liên quan.
- KHÔNG phán ai thắng. KHÔNG bịa số hiệu điều luật cụ thể.
Chỉ trả về đoạn truy vấn, không thêm tiêu đề.

TÓM TẮT: {case_query}

TÌNH TIẾT: {facts}"""

_LABEL_RE = re.compile(r"^\s*truy\s*vấn\s*pháp\s*lý\s*:?\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _default_llm(*args, **kwargs):
    from ..agents.llm import llm
    return llm(*args, **kwargs)


def _clean_output(text: str) -> str:
    text = _LABEL_RE.sub("", str(text or "").strip())
    return _WS_RE.sub(" ", text).strip()


def build_legal_query(case_id: str, case_query: str, facts_text: str, llm_fn=None) -> dict:
    llm_fn = llm_fn or _default_llm
    cq = _clean_case_query(case_query)
    prompt = _PROMPT.format(case_query=cq, facts=facts_text or "(không có tình tiết bổ sung)")
    raw = llm_fn(
        prompt,
        system=_SYSTEM,
        temperature=get("legal_query.temperature", 0.3),
        max_tokens=get("legal_query.max_tokens", 1500),
    )
    legal_query = _clean_output(raw)
    # embed_text = case_query + legal_query — cấu hình THẮNG (R@20 22.9% ở eval BM25).
    # Đây là trường EMBED THẲNG vào bge-m3/Qdrant; retrieval_text giữ = legal_query để
    # eval so A/B tách bạch (case_query cộng riêng).
    embed_text = (cq + " " + legal_query).strip() if cq else legal_query
    return {
        "case_id": case_id,
        "retrieval_text": legal_query,
        "embed_text": embed_text,
        "fields": {"query": cq, "legal_query": legal_query},
        "prompt_version": PROMPT_VERSION,
        "provenance": {"source": "legal_query_rewrite", "has_case_query": bool(cq),
                       "has_facts": bool(facts_text)},
    }


def _load_denoised(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(c.get("case_id")): str(c.get("retrieval_text") or "")
        for c in data.get("cases", []) if isinstance(c, dict)
    }


def _save(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_cases": len(cases), "cases": cases},
                               ensure_ascii=False, indent=2), encoding="utf-8")


def run_all(denoised: dict[str, str], case_ids: list[str], out_path: Path,
            *, fresh: bool = False, llm_fn=None) -> list[dict]:
    results: list[dict] = []
    if out_path.exists() and not fresh:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        results = [c for c in prev.get("cases", [])
                   if isinstance(c, dict) and (c.get("retrieval_text") or "").strip()
                   and c.get("prompt_version") == PROMPT_VERSION]
    done = {str(c.get("case_id")) for c in results}
    for case_id in case_ids:
        if case_id in done:
            continue
        try:
            cq = raw_case(case_id).get("case_query", "")
        except KeyError:
            cq = ""
        result = build_legal_query(case_id, cq, denoised.get(case_id, ""), llm_fn=llm_fn)
        results.append(result)
        _save(out_path, results)   # lưu tăng dần -> resume khi tunnel rớt
        print(f"  ✓ {case_id}: {len(result['retrieval_text'])} chars", file=sys.stderr, flush=True)
    _save(out_path, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal-query rewrite -> chunk embedding retrieve luật")
    parser.add_argument("--denoised", default="outputs/retrieval_chunks.json")
    parser.add_argument("--out", default="outputs/retrieval_chunks_llm.json")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--all", action="store_true", help="Chạy toàn bộ (mặc định khi không có --case)")
    parser.add_argument("--case", dest="case_id")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    denoised = _load_denoised(Path(args.denoised))
    from ..data_io import list_case_ids
    case_ids = [args.case_id] if args.case_id else list_case_ids()
    results = run_all(denoised, case_ids, Path(args.out), fresh=args.fresh)
    print(f"[saved] {args.out} · {len(results)} vụ", file=sys.stderr)


if __name__ == "__main__":
    main()
