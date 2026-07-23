"""
ALQAC 2026 — Case Evidence Retrieval Agent v4
=============================================
Adaptive retrieval + strategic LLM: entity refinement, plateau escape, coverage assess.
LLM used at 3 strategic points only — NOT per-turn.
"""

import re
import sys
import json
import time
import requests
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, TypedDict, Literal

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

sys.stdout.reconfigure(encoding='utf-8')

# ── Config ────────────────────────────────────────────────────────────────────

RETRIEVE_URL = "https://alqac-api.ngrok.pro/retrieve"
RATE_SLEEP   = 5.2

MAX_CALLS       = 50
DUP_WINDOW      = 10
DUP_STOP_RATIO  = 0.75
MIN_COVERAGE    = 0.50
MIN_CALLS_SOFT  = 20
MAX_QUERIES     = 60
MAX_LLM_REFUELS = 5     # lượt gọi LLM sinh query mới tối đa mỗi case (không phải one-shot)

# LLM config
LLM_BASE     = os.getenv("LLM_BASE", "http://localhost:8000/v1")
LLM_MODEL    = os.getenv("LLM_MODEL", "qwen2.5-7b-instruct")
LLM_KEY      = os.getenv("LLM_KEY", "none")

# ── Types ─────────────────────────────────────────────────────────────────────

class SeedQueryInput(TypedDict):
    persons: list[str]
    orgs: list[str]
    amounts: list[str]
    locations: list[str]
    dates: list[str]
    specific_items: list[str]
    case_topic: str

class RetrieveResult(TypedDict):
    new_ids: list[str]
    first_id: Optional[str]
    score: float
    is_new: bool

class LLMEntityRefinement(TypedDict):
    persons: list[str]
    orgs: list[str]
    amounts: list[str]
    locations: list[str]
    dates: list[str]
    specific_items: list[str]
    case_topic: str
    reasoning: str

class LLMQueryGen(TypedDict):
    queries: list[str]
    reasoning: str

class LLMCoverageAssess(TypedDict):
    sufficient: bool
    missing_aspects: list[str]
    reasoning: str

# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class AgentRun:
    case_id: str
    case_query: str
    pool: dict[str, dict[str, str | float]] = field(default_factory=dict)
    calls_made: int = 0
    finished: bool = False
    start_time: float = field(default_factory=time.time)
    history: list[dict] = field(default_factory=list)
    entities: SeedQueryInput = field(default_factory=lambda: {
        "persons": [], "orgs": [], "amounts": [], "locations": [],
        "dates": [], "specific_items": [], "case_topic": ""
    })

# ── Constants ─────────────────────────────────────────────────────────────────

VN_UC = r'A-ZẮẰẲẴẸẺẼỀỂỄỆỈĨỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸ'
VN_LC = r'a-zàáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ'

NON_QUERY_WORDS = set([
    "và", "của", "tại", "ở", "là", "cho", "để", "bị", "được", "các", "những",
    "một", "hai", "ba", "với", "trong", "ngoài", "ra", "vào", "có", "đã", "đang",
    "không", "sẽ", "này", "kia", "đó", "như", "vậy", "thì", "nếu", "vì", "nên",
    "từ", "cũng", "rất", "nhiều", "ít", "mà", "lên", "xuống", "qua", "lại",
    "về", "theo", "sau", "trước", "khi", "cùng", "hay", "hoặc",
    "vụ", "án", "tranh", "chấp", "yêu", "cầu", "bồi", "thường", "thiệt", "hại",
    "hợp", "đồng", "phân", "đoạn", "tìm", "truy", "xuất", "thông", "tin",
    "bản", "tòa", "xét", "xử", "đương", "sự", "khởi", "kiện", "trình", "bày",
    "dự", "đoán", "thắng", "kiện", "nguyên", "đơn", "ai", "được",
    "ông", "bà", "chị", "anh", "cụ", "bác", "cô", "chú", "dì",
    "số", "năm", "tháng", "ngày", "giờ",
    "nhưng", "còn", "mới", "vẫn", "chỉ", "đều", "phải", "nhất",
    "thu", "chi", "thứ", "loại", "việc", "phần", "nơi", "khoản",
    # Vietnamese spelled-out numbers — noise in money/date phrases (e.g. "sáu triệu năm trăm...")
    "bốn", "sáu", "bảy", "tám", "chín", "mười", "mươi", "trăm", "nghìn",
    "triệu", "tỷ", "lăm", "linh", "mốt", "tư", "rưỡi",
    # Generic legal boilerplate — appears in nearly every case, no discriminative value
    "buộc", "chịu", "luật", "pháp", "cộng", "tổng", "cuộc",
    # Dataset template artifact — case_query in ~37/50 test cases ends with a fixed
    # "Agent dự đoán ai được chấp nhận yêu cầu?" suffix; "agent" is English/generic
    # and never appears inside actual case content, so it only ever wastes a call.
    "agent",
])

LEGAL_BIGRAMS = {
    "chuyển nhượng", "quyền sử dụng", "sử dụng đất", "giấy chứng nhận",
    "bồi thường", "thiệt hại", "thương tích", "hợp đồng tín dụng",
    "hợp đồng vay", "thế chấp", "trả nợ", "lãi suất", "thừa kế",
    "chia di sản", "di chúc", "hàng thừa kế", "lấn chiếm", "ranh giới",
    "tài sản chung", "ly hôn", "nuôi con", "cấp dưỡng",
    "bản di chúc", "căn nhà", "căn hộ", "quyền sở hữu",
    "ngân hàng", "công ty", "thư cấp", "tiêu thụ", "chuyển nhượng",
}

TOPIC_QUERIES = {
    "tort": ["bồi thường", "thiệt hại", "thương tích", "tai nạn", "chi phí"],
    "land": ["đất", "thửa", "bản đồ", "giấy chứng nhận", "lấn chiếm", "ranh"],
    "inheritance": ["di sản", "thừa kế", "di chúc", "chia", "hàng thừa kế"],
    "credit": ["vay", "nợ", "hợp đồng tín dụng", "lãi suất", "thế chấp", "trả nợ"],
    "divorce": ["ly hôn", "nuôi con", "cấp dưỡng", "tài sản chung"],
    "property": ["tài sản", "chiếm hữu", "sở hữu", "đòi lại"],
}

PROCEDURAL = ["trình bày", "khởi kiện", "yêu cầu", "phản tố", "chứng cứ"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(rf'[{VN_UC}{VN_LC}]+', text.lower())

def _extract_legal_phrases(text: str, known_terms: set[str]) -> list[str]:
    tokens = _tokenize(text)
    phrases = []
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in LEGAL_BIGRAMS and bg not in known_terms:
            phrases.append(bg)
            known_terms.add(bg)
    i = 0
    while i < len(tokens) - 2:
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        if any(w in NON_QUERY_WORDS or len(w) < 3 for w in (w1, w2, w3)):
            i += 1
            continue
        tg = f"{w1} {w2} {w3}"
        if tg not in known_terms:
            phrases.append(tg)
            known_terms.add(tg)
            i += 3  # skip past this window instead of sliding by 1 → avoid near-duplicate overlapping trigrams
        else:
            i += 1
    return phrases[:5]

# ── LLM Helpers ───────────────────────────────────────────────────────────────

def _llm_chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 1024) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    if LLM_KEY and LLM_KEY != "none":
        headers["Authorization"] = f"Bearer {LLM_KEY}"

    # Tắt reasoning của Qwen3-style models — nếu không, model đốt hết max_tokens
    # cho chain-of-thought (reasoning_content) và content trả về None/rỗng.
    json_payload = {
        "model": LLM_MODEL, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    def _extract(data: dict) -> str:
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        return (reasoning + "\n" + content) if (reasoning and not content) else content

    try:
        r = requests.post(f"{LLM_BASE}/chat/completions", headers=headers, json=json_payload, timeout=120)
        if r.status_code == 400:
            json_payload.pop("chat_template_kwargs", None)
            r = requests.post(f"{LLM_BASE}/chat/completions", headers=headers, json=json_payload, timeout=120)
        r.raise_for_status()
        return _extract(r.json())
    except Exception as e:
        if "chat_template_kwargs" in json_payload:
            try:
                json_payload.pop("chat_template_kwargs", None)
                r = requests.post(f"{LLM_BASE}/chat/completions", headers=headers, json=json_payload, timeout=120)
                r.raise_for_status()
                return _extract(r.json())
            except Exception as e2:
                print(f"    [LLM] {e2}")
                return None
        print(f"    [LLM] {e}")
        return None

def _parse_json_response(msg: str, model: type) -> Optional[dict]:
    try:
        # Extract JSON from response
        match = re.search(r"\{.*\}", msg, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        return data
    except Exception:
        return None

def llm_refine_entities(case_query: str, regex_entities: SeedQueryInput) -> SeedQueryInput:
    """LLM refines/augments regex entity extraction."""
    prompt = f"""You are a Vietnamese legal NER expert. Given a case query and preliminary regex extraction, 
improve the entity extraction. Return ONLY valid JSON matching the schema.

Case query:
{case_query}

Regex extraction (preliminary):
{json.dumps(regex_entities, ensure_ascii=False)}

Tasks:
1. Add missing persons (including initials like NVP, NNS)
2. Add missing organizations (full names)
3. Add missing amounts/measurements
4. Add missing locations (admin units: phường, quận, TP...)
5. Add missing dates
6. Add missing specific items (đất, nhà, xe, chó, hợp đồng, thừa kế...)
7. Confirm or correct case_topic (tort/land/inheritance/credit/divorce/property)

Return JSON with exact keys: persons, orgs, amounts, locations, dates, specific_items, case_topic, reasoning"""

    msg = _llm_chat([
        {"role": "system", "content": "You are a Vietnamese legal NER expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ])
    if not msg:
        return regex_entities
    data = _parse_json_response(msg, LLMEntityRefinement)
    if not data:
        return regex_entities
    # Merge: LLM additions + regex (union). LLM đôi khi trả về item là dict/list
    # thay vì string (vd {"name": "..."} thay vì "..."), set() sẽ crash vì dict
    # không hashable — coerce hết về string trước khi dedupe.
    refined = regex_entities.copy()
    for k in ["persons", "orgs", "amounts", "locations", "dates", "specific_items"]:
        if k in data and isinstance(data[k], list):
            cleaned_list = []
            for item in (refined[k] + data[k]):
                if isinstance(item, dict):
                    val = item.get("value") or item.get("text") or item.get("name") or str(item)
                    cleaned_list.append(str(val))
                elif isinstance(item, list):
                    cleaned_list.append(", ".join(str(x) for x in item))
                elif item is not None:
                    cleaned_list.append(str(item))
            refined[k] = list(set(cleaned_list))[:5]
    if data.get("case_topic") in TOPIC_QUERIES:
        refined["case_topic"] = data["case_topic"]
    return refined

def llm_generate_followup_queries(run: AgentRun, plateau_info: dict) -> list[str]:
    """LLM generates creative follow-up queries when plateauing."""
    seen_queries = list(set(h.get("query", "") for h in run.history if h.get("query")))
    pool_summary = [f"{cid}: {info['text'][:150]}..." for cid, info in list(run.pool.items())[:5]]
    
    prompt = f"""Case query: {run.case_query}

Queries already used ({len(seen_queries)}): {seen_queries}

Chunks collected ({len(run.pool)}):
{chr(10).join(pool_summary)}

Plateau info: {json.dumps(plateau_info)}

Generate 5 SHORT BM25 queries (2-4 keywords each) to retrieve NEW case-content chunks.
Rules:
- Each query: 2-4 specific keywords, NOT a full sentence
- Use ENTITY NAMES, amounts, locations, legal terms SPECIFIC to this case
- DO NOT repeat any query from 'already used'
- DO NOT use case_id
- Focus on aspects NOT yet covered in collected chunks
- Output valid JSON: {{\"queries\": [\"q1\", \"q2\", \"q3\", \"q4\", \"q5\"], \"reasoning\": \"...\"}}"""

    msg = _llm_chat([
        {"role": "system", "content": "You are a legal BM25 query expert. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ], temperature=0.5, max_tokens=512)
    
    if not msg:
        return []
    data = _parse_json_response(msg, LLMQueryGen)
    if not data or not isinstance(data.get("queries"), list):
        return []
    # Dedupe against history
    used = {q.strip().lower() for q in seen_queries}
    fresh = [q for q in data["queries"] if q.strip().lower() not in used]
    print(f"    [LLM] Generated {len(fresh)} fresh queries: {fresh}")
    return fresh

def llm_assess_coverage(run: AgentRun, n_est: float, coverage: float) -> LLMCoverageAssess:
    """LLM assesses if coverage is sufficient."""
    pool_summary = [f"{cid}: {info['text'][:200]}..." for cid, info in list(run.pool.items())[:8]]
    
    prompt = f"""Case query: {run.case_query}

Chunks collected ({len(run.pool)} unique, estimated n≈{n_est:.0f}, coverage={coverage:.0%}):
{chr(10).join(pool_summary)}

Assess if we have sufficient evidence to answer the case query. Consider:
- Are key facts (parties, dispute object, legal basis) covered?
- Are both sides' arguments represented?
- Is the core legal issue (tort/land/inheritance/credit) addressed?
- What specific aspects are MISSING?

Return JSON: {{\"sufficient\": true/false, \"missing_aspects\": [\"...\"], \"reasoning\": \"...\"}}"""

    msg = _llm_chat([
        {"role": "system", "content": "You are a Vietnamese legal evidence assessor. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ], temperature=0.2, max_tokens=512)
    
    if not msg:
        return {"sufficient": coverage >= 0.7, "missing_aspects": [], "reasoning": "LLM failed"}
    data = _parse_json_response(msg, LLMCoverageAssess)
    if not data:
        return {"sufficient": coverage >= 0.7, "missing_aspects": [], "reasoning": "Parse failed"}
    return data

# ── Helpers ───────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(rf'[{VN_UC}{VN_LC}]+', text.lower())

def _extract_legal_phrases(text: str, known_terms: set[str]) -> list[str]:
    tokens = _tokenize(text)
    phrases = []
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in LEGAL_BIGRAMS and bg not in known_terms:
            phrases.append(bg)
            known_terms.add(bg)
    i = 0
    while i < len(tokens) - 2:
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        if any(w in NON_QUERY_WORDS or len(w) < 3 for w in (w1, w2, w3)):
            i += 1
            continue
        tg = f"{w1} {w2} {w3}"
        if tg not in known_terms:
            phrases.append(tg)
            known_terms.add(tg)
            i += 3  # skip past this window instead of sliding by 1 → avoid near-duplicate overlapping trigrams
        else:
            i += 1
    return phrases[:5]

# ── Entity Extractor (Regex) ──────────────────────────────────────────────────

def extract_entities_regex(text: str) -> SeedQueryInput:
    e: SeedQueryInput = {
        "persons": [], "orgs": [], "amounts": [], "locations": [],
        "dates": [], "specific_items": [], "case_topic": ""
    }
    lower = text.lower()
    uc = VN_UC
    lc = VN_LC

    # 1. Persons: honorific + name + initials
    name_unit = rf'[{uc}](?:[{lc}]+|[{uc}]?)'
    honorifics = r'(?:Ông|Bà|Chị|Anh|Cụ|Bác|ông|bà|chị|anh|cụ|bác)'
    persons = re.findall(honorifics + r'\s+((?:' + name_unit + r'(?:\s+' + name_unit + r')+))', text)
    initials = re.findall(r'\b([A-ZẮẰẲẴẸẺẼỀỂỄỆỈĨỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸ]{2,4})\b', text)
    for ini in initials:
        if re.search(rf'(?:ông|bà|chị|anh|cụ|vợ|chồng|con|cái|đồng|thừa kế|khởi kiện)\s+{ini}', text, re.IGNORECASE):
            persons.append(ini)
    if not persons:
        persons = re.findall(rf'([{uc}][{lc}]+(?:\s+[{uc}][{lc}]+)+)', text)
    e["persons"] = list(set(p.strip() for p in persons if len(p) > 2))[:5]

    # 2. Organizations: full name up to connector
    orgs = re.findall(
        r'(Ngân hàng\s+[' + uc + r'][' + lc + r'][\w\s]+?)(?=\s+(?:và|tranh|khởi|về|$))',
        text
    )
    orgs += re.findall(
        r'(Công ty\s+(?:TNHH|CP|Cổ phần)?\s*[' + uc + r'][' + lc + r'][\w\s]+?)(?=\s+(?:và|tranh|khởi|về|$))',
        text
    )
    e["orgs"] = list(set(o.strip() for o in orgs))[:3]

    # 3. Amounts + measurements
    amounts = re.findall(r'(\d[\d.,]*\s*(?:triệu|tỷ|nghìn|đồng|VND|vnd|m2|m²|ha))', text, re.IGNORECASE)
    e["amounts"] = [a.strip() for a in amounts[:5]]

    # 4. Locations
    locs = re.findall(r'((?:phường|xã|thị trấn|quận|huyện|tỉnh|thành phố)\s+[\d\w]+)', text, re.IGNORECASE)
    locs += re.findall(rf'(TP\s+[{uc}][{lc}]+(?:\s+[{uc}][{lc}]+)+)', text)
    e["locations"] = list(set(l.strip() for l in locs))[:3]

    # 5. Dates
    dates = re.findall(r'(ngày\s+\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text, re.IGNORECASE)
    if not dates:
        dates = re.findall(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', text)
    e["dates"] = dates[:3]

    # 6. Specific items
    item_map = {
        "đất": "đất", "nhà": "nhà", "xe": "xe", "chó": "chó",
        "xe máy": "xe máy", "sổ đỏ": "sổ đỏ",
        "giấy chứng nhận": "giấy chứng nhận", "hợp đồng": "hợp đồng",
        "chuyển nhượng": "chuyển nhượng", "thế chấp": "thế chấp",
        "tín dụng": "tín dụng", "thừa kế": "thừa kế", "di sản": "di sản",
    }
    for kw, item in item_map.items():
        if kw in text.lower():
            e["specific_items"].append(item)
    e["specific_items"] = list(set(e["specific_items"]))

    # 7. Case topic
    topic_map = [
        ("bồi thường thiệt hại ngoài hợp đồng", "tort"),
        ("bồi thường thiệt hại", "tort"),
        ("thừa kế", "inheritance"),
        ("chia di sản", "inheritance"),
        ("hợp đồng tín dụng", "credit"),
        ("vay", "credit"),
        ("ly hôn", "divorce"),
        ("quyền sử dụng đất", "land"),
        ("chuyển nhượng", "land"),
        ("lấn chiếm", "land"),
    ]
    for kw, topic in topic_map:
        if kw in text.lower():
            e["case_topic"] = topic
            break

    # Filter locations out of persons
    loc_tokens = set()
    for loc in e["locations"]:
        loc_tokens.update(loc.split())
    e["persons"] = [p for p in e["persons"] if not any(tok in loc_tokens for tok in p.split())]

    return e

def extract_entities(text: str) -> SeedQueryInput:
    regex_e = extract_entities_regex(text)
    return llm_refine_entities(text, regex_e)

# ── Seed Queries ──────────────────────────────────────────────────────────────

def build_seed_queries(entities: SeedQueryInput, case_query: str) -> list[str]:
    queries = []

    # 1. Persons: full name only (most specific for BM25)
    for p in entities["persons"]:
        queries.append(p)

    # 2. Orgs: full name only (short form often too generic)
    for o in entities["orgs"]:
        queries.append(o)

    # 3. Amounts: numeric part
    for a in entities["amounts"]:
        queries.append(a.split()[0])

    # 4. Locations: full admin unit
    for l in entities["locations"]:
        queries.append(l)

    # 5. Dates: date only
    for d in entities.get("dates", []):
        queries.append(d.replace("ngày ", "").split()[0])

    # 6. Topic queries (exclude items already in specific_items)
    topic = entities.get("case_topic", "")
    if topic in TOPIC_QUERIES:
        specific_set = set(entities["specific_items"])
        for q in TOPIC_QUERIES[topic]:
            if q not in specific_set:
                queries.append(q)

    # 7. Specific items (property types, legal docs)
    queries.extend(entities["specific_items"])

    # 8. Procedural (only if not already covered)
    procedural_unique = [q for q in PROCEDURAL[:3] if q not in queries]
    queries.extend(procedural_unique)

    # 9. Rare words from case_query: ALL UPPER abbrev (2-4 chars) OR lowercase >=5 chars
    seen_w = set()
    for m in re.finditer(rf'[{VN_UC}{VN_LC}]+', case_query):
        w = m.group()
        wl = w.lower()
        if wl in NON_QUERY_WORDS or len(wl) < 3:
            continue
        # Keep: ALL UPPER abbrev (2-4 chars) OR lowercase >=5 chars
        if (w.isupper() and 2 <= len(w) <= 4) or (not w.isupper() and len(wl) >= 5):
            if wl not in seen_w:
                seen_w.add(wl)
                queries.append(w)

    # Final dedup (case-insensitive, keep first occurrence = highest priority)
    seen = set()
    deduped = []
    for q in queries:
        qs = q.strip().lower()
        if qs and len(qs) > 1 and qs not in seen:
            seen.add(qs)
            deduped.append(q.strip())
    return deduped

# ── Chunk-Derived Queries ────────────────────────────────────────────────────

def extract_novel_terms(text: str, known_terms: set[str]) -> list[str]:
    phrases = _extract_legal_phrases(text, known_terms)
    # Từ đơn vẫn được thêm dù đã có phrase — nhưng ít hơn hẳn (2 thay vì 5) vì
    # với NON_QUERY_WORDS đã lọc số đếm/từ pháp lý sáo rỗng, một vài từ hiếm còn
    # sót lại (tên riêng, thuật ngữ đặc thù) đôi khi "ăn may" trúng chunk khác mà
    # phrase curated không phủ tới — giữ lại chút diện tích khám phá đó thay vì
    # cắt hẳn về 0, để estimate_n() (tự tham chiếu vào unique đã tìm được) không
    # bị bó hẹp sớm và co cụm "vùng miễn phí" của should_stop() quá nhanh.
    take = 2 if phrases else 5
    terms = set()
    for w in _tokenize(text):
        if w not in NON_QUERY_WORDS and w not in known_terms and len(w) >= 3:
            terms.add(w)
    sorted_terms = sorted(terms, key=lambda t: (-len(t), t))[:take]
    known_terms.update(sorted_terms)
    return phrases + sorted_terms

def _extract_legal_phrases(text: str, known_terms: set[str]) -> list[str]:
    tokens = _tokenize(text)
    phrases = []
    for i in range(len(tokens) - 1):
        bg = f"{tokens[i]} {tokens[i+1]}"
        if bg in LEGAL_BIGRAMS and bg not in known_terms:
            phrases.append(bg)
            known_terms.add(bg)
    i = 0
    while i < len(tokens) - 2:
        w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
        if any(w in NON_QUERY_WORDS or len(w) < 3 for w in (w1, w2, w3)):
            i += 1
            continue
        tg = f"{w1} {w2} {w3}"
        if tg not in known_terms:
            phrases.append(tg)
            known_terms.add(tg)
            i += 3  # skip past this window instead of sliding by 1 → avoid near-duplicate overlapping trigrams
        else:
            i += 1
    return phrases[:5]

# ── Retrieval ─────────────────────────────────────────────────────────────────

def exec_retrieve(case_id: str, query: str, token: str, run: AgentRun) -> RetrieveResult:
    max_retries = 3
    sleep_time = RATE_SLEEP
    results = []
    t_start = time.time()

    for attempt in range(max_retries):
        try:
            r = requests.post(
                RETRIEVE_URL,
                headers={"x-api-key": token, "Content-Type": "application/json"},
                json={"case_id": case_id, "query": query},
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(sleep_time)
                sleep_time *= 2
                continue
            r.raise_for_status()
            results = r.json().get("results", [])
            break
        except requests.exceptions.RequestException as req_err:
            if attempt == max_retries - 1:
                run.calls_made += 1
                run.history.append({"turn": run.calls_made, "query": query, "chunk_id": None, "is_new": False})
                return {"new_ids": [], "first_id": None, "score": 0.0, "is_new": False}
            time.sleep(sleep_time)
            sleep_time *= 2

    latency = time.time() - t_start
    run.calls_made += 1
    time.sleep(RATE_SLEEP)

    if not results:
        run.history.append({"turn": run.calls_made, "query": query, "chunk_id": None, "is_new": False})
        return {"new_ids": [], "first_id": None, "score": 0.0, "is_new": False}

    new_ids = []
    first_id = None
    first_score = 0.0

    for i, hit in enumerate(results):
        cid = hit["chunk_id"]
        score = hit.get("score", 0)
        txt = hit["text"]
        is_new = cid not in run.pool
        if is_new:
            run.pool[cid] = {"text": txt, "score": score, "query": query}
            new_ids.append(cid)
        if i == 0:
            first_id = cid
            first_score = score

    run.history.append({
        "turn": run.calls_made, "query": query, "chunk_id": first_id,
        "is_new": bool(new_ids), "score": first_score, "latency": latency,
    })

    return {"new_ids": new_ids, "first_id": first_id, "score": first_score, "is_new": bool(new_ids)}

# ── Stop Logic ────────────────────────────────────────────────────────────────

def estimate_n(unique: int, calls: int) -> float:
    return max(20.0, unique * 2.0)

def should_stop(dup_window: deque, calls: int, unique: int) -> bool:
    if calls >= MAX_CALLS:
        return True
    if len(dup_window) == DUP_WINDOW:
        dup_ratio = sum(dup_window) / DUP_WINDOW
        n_est = estimate_n(unique, calls)
        coverage = unique / max(n_est, 1)
        in_free_zone = calls < 2 * n_est
        if dup_ratio >= DUP_STOP_RATIO and coverage >= MIN_COVERAGE and not in_free_zone:
            return True
        if dup_ratio >= DUP_STOP_RATIO and unique >= 12 and calls >= MIN_CALLS_SOFT and not in_free_zone:
            return True
    return False

# ── Main Agent ────────────────────────────────────────────────────────────────

class RetrievalAgentV4:

    def __init__(self, token: str):
        self.token = token

    def run(self, case_id: str, case_query: str) -> AgentRun:
        run = AgentRun(case_id=case_id, case_query=case_query)
        # Cục bộ theo từng case — tránh rò rỉ "đã dùng LLM" sang case kế tiếp khi
        # chạy batch (--json) với cùng một agent instance.
        llm_refuels_used = 0
        llm_coverage_used = False

        print(f"\n{'='*60}")
        print(f"RETRIEVAL AGENT V4  case={case_id}")
        print(f"{'='*60}")

        # Step 1: Entity extraction (regex + LLM refine)
        print("\n[1/4] Entity extraction (regex + LLM refine)")
        run.entities = extract_entities(case_query)
        print(f"  Persons   : {run.entities['persons']}")
        print(f"  Orgs      : {run.entities['orgs']}")
        print(f"  Amounts   : {run.entities['amounts']}")
        print(f"  Locations : {run.entities['locations']}")
        print(f"  Dates     : {run.entities['dates']}")
        print(f"  Topic     : {run.entities['case_topic']}")

        # Step 2: Seed queries
        print(f"\n[2/4] Seed queries")
        query_queue = build_seed_queries(run.entities, case_query)
        print(f"  {len(query_queue)} seeds: {query_queue[:8]}...")

        # Step 3: Adaptive retrieval
        print(f"\n[3/4] Adaptive retrieval")
        dup_window = deque(maxlen=DUP_WINDOW)
        seen_terms = set()
        plateau_count = 0
        # -1 = chưa escape lần nào -> luôn cho phép lượt escape đầu tiên. Sau đó,
        # chỉ cấp thêm lượt escape nếu escape TRƯỚC thực sự tìm ra chunk mới — LLM
        # gần như luôn sinh được CÂU CHỮ mới (khác text đã dùng), nhưng câu chữ mới
        # không đồng nghĩa với NỘI DUNG mới; nếu chỉ dựa vào "có query mới" để reset
        # plateau_count, case đã cạn nội dung vẫn có thể ăn hết cả 5 lượt refuel
        # (mỗi lượt tự "thành công" về mặt câu chữ) và tốn tới tận MAX_CALLS vô ích.
        unique_at_last_escape = -1

        while run.calls_made < MAX_CALLS:
            if not query_queue:
                stalled = len(run.pool) == unique_at_last_escape
                if not stalled and llm_refuels_used < MAX_LLM_REFUELS:
                    print(f"  [REFUEL] Queue empty — calling LLM for fresh queries ({llm_refuels_used + 1}/{MAX_LLM_REFUELS})...")
                    llm_refuels_used += 1
                    unique_at_last_escape = len(run.pool)
                    plateau_info = {
                        "calls": run.calls_made,
                        "unique": len(run.pool),
                        "dup_ratio": sum(dup_window) / max(len(dup_window), 1),
                        "used_queries": list(set(h.get("query", "") for h in run.history)),
                    }
                    llm_queries = llm_generate_followup_queries(run, plateau_info)
                    existing = set(q.strip().lower() for q in query_queue)
                    fresh = [q for q in llm_queries if q.strip().lower() not in existing
                             and q.strip().lower() not in (h.get("query", "").lower() for h in run.history)]
                    if fresh:
                        query_queue.extend(fresh[:MAX_QUERIES - len(query_queue)])
                        plateau_count = 0
                        print(f"    [LLM] Added {len(fresh)} fresh queries")
                    else:
                        print(f"    [LLM] No fresh queries generated")
                if not query_queue:
                    reason = "escape trước không tìm thêm được chunk nào, case đã cạn" if stalled else "hết nguồn query"
                    print(f"  [OUT OF QUERIES] {reason} — stopping at calls={run.calls_made}")
                    break

            query = query_queue.pop(0)
            c = run.calls_made

            print(f"  [{c}/{MAX_CALLS}] queue={len(query_queue)}  retrieve(\"{query}\")", end="")
            result = exec_retrieve(case_id, query, self.token, run)

            flag = "NEW" if result["is_new"] else "DUP"
            print(f"  \u2192 [{flag}] {result['first_id']}")

            dup_window.append(not result["is_new"])

            if result["is_new"]:
                plateau_count = 0
                for cid in result["new_ids"]:
                    txt = run.pool[cid]["text"][:80].replace("\n", " ").strip()
                    print(f"      NEW {cid}: {txt}...")
                    novel = extract_novel_terms(run.pool[cid]["text"], seen_terms)
                    if novel:
                        seen_terms.update(novel)
                        existing = set(q.strip().lower() for q in query_queue)
                        fresh = [q for q in novel
                                 if q.strip().lower() not in existing
                                 and q.strip().lower() not in (h.get("query","").lower() for h in run.history)]
                        if fresh:
                            query_queue.extend(fresh[:MAX_QUERIES - len(query_queue)])
            else:
                plateau_count += 1

            # Plateau escape: LLM generates fresh queries — chỉ cấp thêm lượt nếu
            # escape TRƯỚC đó thực sự tìm ra chunk mới (không chỉ "có query mới").
            if plateau_count >= DUP_WINDOW:
                made_progress = len(run.pool) > unique_at_last_escape
                if made_progress and llm_refuels_used < MAX_LLM_REFUELS:
                    print(f"  [PLATEAU] {plateau_count} DUPs — calling LLM for fresh queries ({llm_refuels_used + 1}/{MAX_LLM_REFUELS})...")
                    llm_refuels_used += 1
                    unique_at_last_escape = len(run.pool)
                    plateau_info = {
                        "calls": run.calls_made,
                        "unique": len(run.pool),
                        "dup_ratio": sum(dup_window) / max(len(dup_window), 1),
                        "used_queries": list(set(h.get("query","") for h in run.history)),
                    }
                    llm_queries = llm_generate_followup_queries(run, plateau_info)
                    if llm_queries:
                        existing = set(q.strip().lower() for q in query_queue)
                        fresh = [q for q in llm_queries if q.strip().lower() not in existing
                                 and q.strip().lower() not in (h.get("query","").lower() for h in run.history)]
                        if fresh:
                            query_queue.extend(fresh[:MAX_QUERIES - len(query_queue)])
                            plateau_count = 0
                            print(f"    [LLM] Added {len(fresh)} fresh queries, resetting plateau")
                    else:
                        print(f"    [LLM] No fresh queries generated")
                else:
                    # Hoặc đã hết 5 lượt LLM refuel, hoặc lượt escape gần nhất không
                    # tìm thêm được chunk nào dù "thành công" về mặt sinh query mới —
                    # cả 2 đều là bằng chứng case đã cạn nội dung thật sự. Case có
                    # unique quá thấp để chạm nhánh "unique>=12"/"coverage>=50%" của
                    # should_stop() (không bao giờ đạt nếu case chỉ có ít đoạn thật),
                    # nên nếu không có nhánh này, nó sẽ cứ gọi DUP tới tận MAX_CALLS.
                    reason = "escape trước không tìm thêm được chunk nào" if not made_progress \
                        else f"hết {MAX_LLM_REFUELS}/{MAX_LLM_REFUELS} lượt LLM refuel"
                    print(f"  [STALLED] {plateau_count} DUPs liên tiếp, {reason} — case đã cạn, dừng sớm ở calls={run.calls_made}.")
                    break

            # Coverage assessment (once per case, near stop)
            n_est = estimate_n(len(run.pool), run.calls_made)
            coverage = len(run.pool) / max(n_est, 1)
            if (coverage >= MIN_COVERAGE and run.calls_made >= MIN_CALLS_SOFT
                and not llm_coverage_used):
                print(f"  [COVERAGE] {coverage:.0%} — LLM assessment...")
                llm_coverage_used = True
                assessment = llm_assess_coverage(run, estimate_n(len(run.pool), run.calls_made), 
                                                 len(run.pool) / max(estimate_n(len(run.pool), run.calls_made), 1))
                print(f"    [LLM] Sufficient: {assessment['sufficient']}, Missing: {assessment['missing_aspects']}")
                if assessment["sufficient"]:
                    print(f"  [LLM STOP] Coverage sufficient per LLM")
                    break

            # Stop check
            if should_stop(dup_window, run.calls_made, len(run.pool)):
                n_est = estimate_n(len(run.pool), run.calls_made)
                coverage = len(run.pool) / max(n_est, 1)
                print(f"  [STOP] calls={run.calls_made}, unique={len(run.pool)}, n_est={n_est:.1f}, coverage={coverage:.0%}")
                break

        # Step 4: Summary
        print(f"\n[4/4] Done")
        evidence = sorted(run.pool.keys())
        print(f"  Calls: {run.calls_made}  Unique: {len(evidence)}")
        print(f"\n  ── Evidence ({len(evidence)} chunks) ──")
        for cid in evidence:
            info = run.pool[cid]
            print(f"  [{cid}] (query=\"{info.get('query', '')}\", score={info.get('score', 0):.2f})")
            print(f"    {info['text']}")
        run.finished = True
        return run

# ── Entry Point ──────────────────────────────────────────────────────────────

def _build_result_entry(cid: str, run: AgentRun, dur: float) -> dict:
    n_est = estimate_n(len(run.pool), run.calls_made)
    E_i = max(0.0, 1.0 - max(0, run.calls_made - 2 * n_est) / (3 * n_est))
    coverage = len(run.pool) / max(n_est, 1)
    score = coverage * E_i
    evidence_ids = sorted(run.pool.keys())
    evidence_details = [
        {
            "chunk_id": eid,
            "query": run.pool[eid].get("query", ""),
            "score": run.pool[eid].get("score", 0),
            "text": run.pool[eid]["text"],
        }
        for eid in evidence_ids
    ]
    return {
        "case_id": cid, "calls": run.calls_made, "unique": len(run.pool),
        "evidence": evidence_ids, "evidence_details": evidence_details, "duration": dur,
        "n_est": round(n_est, 1), "E_i": round(E_i, 3),
        "coverage_pct": round(coverage * 100, 1), "score": round(score, 3),
    }


def _save_result(results_path: str, entry: dict) -> None:
    """Merge entry vào file kết quả, thay thế entry cũ cùng case_id nếu có."""
    existing = []
    if os.path.exists(results_path):
        try:
            with open(results_path, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing = [r for r in existing if r.get("case_id") != entry["case_id"]]
    existing.append(entry)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def _load_case_query(case_id: str) -> Optional[str]:
    path = os.path.join(os.path.dirname(__file__ or "."), "ALQAC2026_public_test.json")
    if os.path.exists(path):
        try:
            cases = json.load(open(path, encoding="utf-8"))
            for c in cases:
                if c["case_id"] == case_id:
                    return c["case_query"]
        except Exception:
            pass
    return None

if __name__ == "__main__":
    if not os.getenv("ALQAC_TOKEN"):
        print("ALQAC_TOKEN not set in .env")
        sys.exit(1)
    ALQAC_TOKEN = os.getenv("ALQAC_TOKEN", "")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", type=str)
    parser.add_argument("--case-query", type=str, help="If omitted, looked up from ALQAC2026_public_test.json")
    parser.add_argument("--json", type=str, help="Path to test JSON file")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=str, default="agent_v4_results.json",
                        help="Path to results JSON file (resume/ghi kết quả)")
    args = parser.parse_args()
    agent = RetrievalAgentV4(token=ALQAC_TOKEN)

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            cases = json.load(f)[:args.limit]

        results_path = args.out
        results = []
        if os.path.exists(results_path):
            try:
                with open(results_path, encoding="utf-8") as f:
                    results = json.load(f)
            except Exception:
                results = []
        # Resumable: case đã có kết quả THÀNH CÔNG (không lỗi) thì bỏ qua, còn lại
        # (chưa chạy, hoặc lần trước bị [CASE FAILED]) sẽ được chạy/thử lại.
        done_ids = {r["case_id"] for r in results if not r.get("error")}

        for i, case in enumerate(cases):
            cid = case["case_id"]
            cq = case["case_query"]
            if cid in done_ids:
                print(f"\n[{i+1}/{len(cases)}] {cid}: đã có kết quả — bỏ qua")
                continue
            print(f"\n{'#'*60}")
            print(f"# CASE {i+1}/{len(cases)}: {cid}")
            print(f"{'#'*60}")
            t0 = time.time()
            try:
                run = agent.run(cid, cq)
            except Exception as e:
                # Không để 1 case lỗi (vd LLM trả JSON dị dạng) giết cả batch nhiều
                # tiếng đồng hồ — log lỗi, ghi nhận case này thất bại, rồi sang case kế.
                print(f"  [CASE FAILED] {cid}: {type(e).__name__}: {e}")
                entry = {
                    "case_id": cid, "calls": None, "unique": 0,
                    "evidence": [], "evidence_details": [], "duration": round(time.time() - t0, 2),
                    "n_est": None, "E_i": None, "coverage_pct": None, "score": 0.0,
                    "error": f"{type(e).__name__}: {e}",
                }
            else:
                dur = round(time.time() - t0, 2)
                entry = _build_result_entry(cid, run, dur)
            results = [r for r in results if r["case_id"] != cid]
            results.append(entry)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE: {len(results)} cases")
        for r in sorted(results, key=lambda r: r["case_id"]):
            if r.get("error"):
                print(f"  {r['case_id']}: [FAILED] {r['error']}")
            else:
                print(f"  {r['case_id']}: {r['calls']} calls → {r['unique']} uniq ({r['coverage_pct']:.0f}% cov, E_i={r['E_i']:.2f}, score={r['score']:.2f}), {r['duration']}s")
    elif args.case_id:
        cq = args.case_query or _load_case_query(args.case_id)
        if not cq:
            print(f"case_query not found for {args.case_id}")
            sys.exit(1)
        t0 = time.time()
        run = agent.run(args.case_id, cq)
        dur = round(time.time() - t0, 2)
        entry = _build_result_entry(args.case_id, run, dur)
        print(f"\nDone in {dur}s. {entry['unique']} unique / {entry['calls']} calls")
        print(f"  n_est={entry['n_est']}, coverage={entry['coverage_pct']}%, E_i={entry['E_i']}, score={entry['score']}")

        results_path = args.out
        _save_result(results_path, entry)
        print(f"  Saved full results (incl. chunk text) to {results_path}")
    else:
        parser.print_help()