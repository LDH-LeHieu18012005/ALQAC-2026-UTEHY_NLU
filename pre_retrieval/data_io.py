"""
data_io — nạp corpus luật + tập test 1 lần, dựng các bảng tra cứu dùng chung.

Sự thật dữ liệu (đã kiểm chứng từ file thật):
  * corpus = 18 văn bản, ~3352 điều. Mỗi điều = {aid, content_Article}.
    `aid` là id nguyên TOÀN CỤC, KHÔNG phải số điều.
  * "Số Điều" KHÔNG được ghi tường minh -> số điều = VỊ TRÍ (điều thứ N).
    (kiểm chứng: Điều 584 của 91/2015/QH13 -> aid 53354).
  * Khoá định danh để nộp = cặp (law_id, aid).

Bảng dựng ra:
  AID2META[(law_id, aid)] -> {content_Article, article_no, corpus_id, procedural}
  KEY2AID[(law_id, article_no)] -> aid           # để map gold "Tên luật | Điều N"
  ARTICLES: list các Article (phục vụ BM25/index)
"""
from __future__ import annotations

import json
import re
from functools import lru_cache

from .config import CFG, get

# Văn bản THỦ TỤC có trong corpus — KHÔNG search ngữ nghĩa (xem law_procedural).
PROCEDURAL_LAWS = {"92/2015/QH13", "93/2015/QH13", "326/2016/UBTVQH14", "26/2008/QH12"}


class Article:
    __slots__ = ("aid", "law_id", "article_no", "text", "corpus_id")

    def __init__(self, aid, law_id, article_no, text, corpus_id):
        self.aid, self.law_id, self.article_no = int(aid), law_id, article_no
        self.text, self.corpus_id = text, corpus_id

    @property
    def is_procedural(self) -> bool:
        return self.law_id in PROCEDURAL_LAWS

    @property
    def ref(self) -> str:
        return f"{self.law_id} | Điều {self.article_no}"

    @property
    def key(self) -> tuple:
        """Khoá định danh để nộp: (law_id, aid)."""
        return (self.law_id, self.aid)

    def evidence(self) -> dict:
        return {"law_id": self.law_id, "aid": self.aid}


@lru_cache(maxsize=1)
def load_corpus() -> dict:
    docs = json.loads(CFG["paths"]["corpus_law"].read_text(encoding="utf-8"))
    by_aid, by_law_article, articles, aid2meta = {}, {}, [], {}
    for doc in docs:
        law_id, corpus_id = doc["law_id"], doc["id"]
        for i, art in enumerate(doc["content"]):
            a = Article(art["aid"], law_id, i + 1, art["content_Article"], corpus_id)
            by_aid[a.aid] = a
            by_law_article[(law_id, a.article_no)] = a.aid          # KEY2AID
            aid2meta[(law_id, a.aid)] = {
                "content_Article": a.text, "article_no": a.article_no,
                "corpus_id": corpus_id, "procedural": a.is_procedural,
            }
            articles.append(a)
    return {"by_aid": by_aid, "KEY2AID": by_law_article,
            "AID2META": aid2meta, "ARTICLES": articles}


def AID2META(law_id: str, aid: int) -> dict | None:
    return load_corpus()["AID2META"].get((law_id, int(aid)))


def get_article(law_id: str, article_no: int) -> Article | None:
    aid = load_corpus()["KEY2AID"].get((law_id, article_no))
    return load_corpus()["by_aid"].get(aid) if aid is not None else None


def article_by_aid(aid: int) -> Article | None:
    return load_corpus()["by_aid"].get(int(aid))


def substantive_articles() -> list[Article]:
    return [a for a in load_corpus()["ARTICLES"] if not a.is_procedural]


# ===========================================================================
# Map gold "Tên luật | Điều N" -> (law_id, aid).
#   gold ghi tên luật dạng tự do (~70 cách viết) + nhiều phiên bản NGOÀI corpus
#   (BLDS 2005/1995, Luật đất đai cũ, án lệ, pháp lệnh...). Đó là TRẦN CỨNG:
#   trả None -> đếm là "ngoài corpus" khi chấm Law-F1, không phải lỗi mô hình.
# ===========================================================================
# Quy tắc nhận diện: (regex tên, năm bắt buộc | None) -> law_id trong corpus.
# Thứ tự QUAN TRỌNG: rule cụ thể (có năm) đặt trước rule chung.
_NAME_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"tố tụng dân sự", re.I), "92/2015/QH13"),   # BLTTDS chỉ có 2015 trong corpus
    (re.compile(r"tố tụng hành chính", re.I), "93/2015/QH13"),
    (re.compile(r"dân sự.*(2015|năm 2015)", re.I), "91/2015/QH13"),
    (re.compile(r"(2015|năm 2015).*dân sự", re.I), "91/2015/QH13"),
    (re.compile(r"326/2016", re.I), "326/2016/UBTVQH14"),
    (re.compile(r"đất đai.*2013|2013.*đất đai|đất đai năm 2013", re.I), "45/2013/QH13"),
    (re.compile(r"hôn nhân.*(2014|năm 2014)", re.I), "52/2014/QH13"),
    (re.compile(r"xây dựng.*(2014|năm 2014)", re.I), "50/2014/QH13"),
    (re.compile(r"tổ chức tín dụng", re.I), "47/2010/QH12"),
    (re.compile(r"kinh doanh bất động sản.*(2014|năm 2014)", re.I), "66/2014/QH13"),
    (re.compile(r"thi hành án dân sự", re.I), "26/2008/QH12"),
    (re.compile(r"hộ tịch", re.I), "60/2014/QH13"),
    (re.compile(r"hình sự.*(2015|năm 2015)", re.I), "100/2015/QH13"),
    (re.compile(r"37/2015", re.I), "37/2015/NĐ-CP"),
    # "Bộ luật dân sự" trần (không năm) -> mặc định bản 2015 trong corpus.
    (re.compile(r"bộ luật dân sự(?!.*200[05])(?!.*1995)", re.I), "91/2015/QH13"),
]

# Phiên bản NGOÀI corpus — nhận diện để loại sớm (log là trần cứng).
_OUT_OF_CORPUS = re.compile(
    r"(dân sự.*(2005|1995|năm 2005|năm 1995)|(2005|1995).*dân sự|"
    r"đất đai.*(2003|1987)|án lệ|pháp lệnh|hôn nhân.*2000|kinh doanh bất động sản.*2006)",
    re.I,
)


def resolve_law_id(name: str) -> str | None:
    """Tên luật tự do -> law_id trong corpus, hoặc None nếu ngoài corpus."""
    name = name.strip()
    if _OUT_OF_CORPUS.search(name):
        return None
    for pat, law_id in _NAME_RULES:
        if pat.search(name):
            return law_id
    return None


_GOLD_LINE = re.compile(r"^(.*?)\|\s*Điều\s+(\d+)", re.I)


def parse_gold_provisions(raw: str) -> dict:
    """
    Tách `related_law_provisions` (text nhiều dòng "Tên luật | Điều N").
    Trả về:
      keys:        set((law_id, aid))       # các trích dẫn ánh xạ được vào corpus
      out_of_corpus: list[str]              # các dòng KHÔNG có trong corpus (trần cứng)
    """
    keys, out = set(), []
    key2aid = load_corpus()["KEY2AID"]
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _GOLD_LINE.match(line)
        if not m:
            continue
        law_id = resolve_law_id(m.group(1))
        art_no = int(m.group(2))
        aid = key2aid.get((law_id, art_no)) if law_id else None
        if aid is not None:
            keys.add((law_id, aid))
        else:
            out.append(line)
    return {"keys": keys, "out_of_corpus": out}


# ===========================================================================
# Tập test: nạp theo case_id. (leak guard nằm ở guardrails.py — đây chỉ là I/O thô.)
# ===========================================================================
@lru_cache(maxsize=1)
def _raw_test_index() -> dict:
    cases = json.loads(CFG["paths"]["public_test"].read_text(encoding="utf-8"))
    return {str(c["case_id"]): c for c in cases}


def list_case_ids() -> list[str]:
    return list(_raw_test_index().keys())


def raw_case(case_id: str) -> dict:
    """Bản ghi THÔ (đầy đủ field). CHỈ guardrails/gold được gọi — không vào prompt."""
    rec = _raw_test_index().get(str(case_id))
    if rec is None:
        raise KeyError(f"case_id {case_id!r} không có trong tập test")
    return rec
