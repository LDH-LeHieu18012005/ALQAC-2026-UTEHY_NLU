"""
agents/llm — wrapper LLM DUY NHẤT cho 3 agent. Backend: HF Inference router
(OpenAI-style /v1/chat/completions), model open-weight < 10B (Qwen2.5-7B mặc định).

Ràng buộc cuộc thi: KHÔNG model thương mại trong runtime; KHÔNG load model local.
Tái lập: temperature thấp + seed cố định + cache đĩa (chạy lại không tốn credit).
"""
from __future__ import annotations

import hashlib
import json
import time

import requests

from ..config import HF_TOKEN, LLM_API_BASE, LLM_API_KEY, NTFY_TOPIC, get
from ..config import CFG

_CACHE_PATH = CFG["paths"]["cache_dir"] / "llm_cache.json"
_cache: dict | None = None

# Ghim ngôn ngữ: Qwen2.5-7B ở nhiệt độ cao dễ trôi sang tiếng Trung.
VI_SYSTEM = ("Bạn là chuyên gia pháp luật dân sự Việt Nam. BẮT BUỘC trả lời HOÀN "
             "TOÀN bằng tiếng Việt, tuyệt đối không dùng tiếng Trung hay ngôn ngữ khác.")


class LLMError(RuntimeError):
    pass


def _load_cache() -> dict:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            _cache = {}
    return _cache


def _cache_key(prompt, system, temperature, max_tokens) -> str:
    raw = json.dumps([get("llm.provider"), get("llm.model"), get("llm.seed"),
                      system, prompt, temperature, max_tokens], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _store(key: str, value: str):
    c = _load_cache()
    c[key] = value
    _CACHE_PATH.write_text(json.dumps(c, ensure_ascii=False), encoding="utf-8")


# Cache URL đã discover (URL Cloudflare đổi mỗi lần chạy notebook; chỉ fetch lại khi cần).
_discovered = {"base": None}


def _discover_ntfy() -> str:
    """Lấy URL vLLM server MỚI NHẤT mà notebook publish lên ntfy.sh/<topic>."""
    if not NTFY_TOPIC:
        raise LLMError("Thiếu NTFY_TOPIC (.env) hoặc llm.ntfy_topic (config) để discover server.")
    try:
        r = requests.get(f"https://ntfy.sh/{NTFY_TOPIC}/json?poll=1", timeout=15)
        latest = None
        for line in r.text.splitlines():
            try:
                m = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = (m.get("message") or "").strip()
            if m.get("event") == "message" and msg.startswith("http"):
                latest = msg            # bản tin sau cùng = URL hiện hành
        if not latest:
            raise LLMError(
                f"ntfy topic '{NTFY_TOPIC}' chưa có URL nào. Hãy chạy hết notebook "
                f"serving/kaggle-qwen-vllm-server.ipynb (Cell 3 publish URL) rồi thử lại.")
        return latest.rstrip("/")
    except requests.RequestException as e:
        raise LLMError(f"Không truy cập được ntfy.sh để discover server: {e}")


def _endpoint(force_discover: bool = False) -> tuple[str, str]:
    """
    (base_url, bearer_token) theo provider. Mọi nhánh đều dùng schema OpenAI
    /v1/chat/completions nên agent KHÔNG cần biết nguồn.
      * openai_compat : vLLM trên notebook (auto-discover qua ntfy) hoặc LLM_API_BASE thủ công.
      * hf            : HF Inference router.
    """
    if get("llm.provider") == "openai_compat":
        if not LLM_API_KEY:
            raise LLMError("provider=openai_compat cần LLM_API_KEY trong .env "
                           "(khớp --api-key của vLLM ở notebook).")
        base = LLM_API_BASE.rstrip("/") if LLM_API_BASE else ""
        if not base and get("llm.discovery", "ntfy") == "ntfy":
            if force_discover or not _discovered["base"]:
                _discovered["base"] = _discover_ntfy()
            base = _discovered["base"]
        if not base:
            raise LLMError("Thiếu base_url: đặt LLM_API_BASE thủ công, hoặc bật discovery=ntfy.")
        return base, LLM_API_KEY
    if not HF_TOKEN:
        raise LLMError("Thiếu HF_TOKEN trong .env")
    return get("llm.api_base").rstrip("/"), HF_TOKEN


def _strip_think(text: str) -> str:
    """
    Qwen3/3.5 mặc định sinh khối <think>...</think> TRƯỚC câu trả lời. vLLM chạy
    --reasoning-parser qwen3 đã tách nó vào reasoning_content (content sạch), nhưng
    nếu server quên bật parser thì content vẫn dính thinking -> tự cắt ở client.
    Trường hợp bị CẮT CỤT giữa chừng thinking (không có </think>) -> trả phần sau
    thẻ mở cuối cùng (đằng nào cũng không có JSON thì extract_json sẽ báo lỗi rõ).
    """
    import re
    if "<think>" not in text:
        return text
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "<think>" in out:                       # thẻ mở KHÔNG đóng (output cắt cụt)
        out = out.rsplit("<think>", 1)[-1].strip()
    return out or text   # content toàn thinking -> trả nguyên văn cho extract_json thử vớt JSON


def _sampling_params(temperature: float) -> dict:
    """
    Tham số sampling theo khuyến nghị CHÍNH THỨC của Qwen3/3.5.
    thinking BẬT: ÉP SÀN temp >= thinking_temperature (Qwen cảnh báo greedy + thinking
      = lặp vô hạn + xuống cấp) + top_p/top_k + presence_penalty chống lặp. Do đó
      judge.temperature=0.0 truyền vào sẽ TỰ được nâng lên (không greedy).
    thinking TẮT: temp giữ nguyên (mặc định ~0.7) + top_p_nonthinking (0.8).
    top_k/presence_penalty: vLLM nhận thẳng trong body (mở rộng ngoài chuẩn OpenAI).
    """
    if get("llm.enable_thinking", True):
        temperature = max(temperature, get("llm.thinking_temperature", 0.6))
        return {"temperature": temperature,
                "top_p": get("llm.top_p", 0.95),
                "top_k": get("llm.top_k", 20),
                "presence_penalty": get("llm.presence_penalty", 1.0)}
    return {"temperature": temperature,
            "top_p": get("llm.top_p_nonthinking", 0.8),
            "top_k": get("llm.top_k", 20)}


def _consume_stream(resp) -> tuple[str, str, str | None]:
    """
    Đọc SSE (stream=True) -> (content, reasoning_content, finish_reason).
    Gom delta.content + delta.reasoning_content qua từng chunk.
    """
    content, reasoning, fin = [], [], None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        raw = raw[5:].strip()
        if raw == "[DONE]":
            break
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        ch = (obj.get("choices") or [{}])[0]
        delta = ch.get("delta") or {}
        if delta.get("content"):
            content.append(delta["content"])
        if delta.get("reasoning_content"):
            reasoning.append(delta["reasoning_content"])
        if ch.get("finish_reason"):
            fin = ch["finish_reason"]
    return "".join(content), "".join(reasoning), fin


def _call_hf(prompt, system, temperature, max_tokens) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    payload = {"model": get("llm.model"), "messages": messages,
               "max_tokens": max_tokens, "seed": get("llm.seed"),
               # STREAMING BẮT BUỘC: tunnel Cloudflare MIỄN PHÍ có timeout ~100s/HTTP
               # request; call thinking Qwen3.5 sinh >100s sẽ bị cắt 524 nếu KHÔNG stream.
               # Khi stream, token chảy liên tục -> Cloudflare không bao giờ hit timeout.
               "stream": True,
               **_sampling_params(temperature)}
    if not get("llm.enable_thinking", True):
        # Tắt thinking của Qwen3/3.5 (nhanh hơn nhưng suy luận nông hơn).
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    last = None
    # KIÊN NHẪN chờ khi server CHẾT (notebook Kaggle hết session / tunnel rớt giữa vụ):
    # thay vì bỏ sau 3 lần thử ~10s, chờ tới server_down_retry_seconds, TỰ dò URL mới
    # từ ntfy để bạn kịp CHẠY LẠI notebook -> pipeline TIẾP TỤC, không mất vụ đang chạy.
    down_deadline = time.time() + get("llm.server_down_retry_seconds", 600)
    poll = get("llm.server_down_poll_seconds", 15)
    short_left = get("llm.retries", 3)      # số lần thử cho lỗi KHÁC (không phải server chết)
    announced = False
    while True:
        server_down = False
        try:
            # _endpoint tự discover lại khi _discovered["base"]=None (đã bị xoá do server chết).
            base, token = _endpoint(force_discover=False)
            url = f"{base}/v1/chat/completions"
            headers = {"Authorization": f"Bearer {token}"}
            # stream=True: timeout dạng (connect, read-giữa-2-chunk). Token chảy đều nên
            # read-timeout chỉ kích hoạt khi server IM > ngưỡng (thật sự treo), KHÔNG
            # phụ thuộc tổng thời gian sinh -> call thinking dài bao lâu cũng không sao.
            resp = requests.post(url, json=payload, headers=headers, stream=True,
                                 timeout=(get("llm.connect_timeout", 15),
                                          get("llm.stream_read_timeout", 120)))
            status = resp.status_code
            if status == 200:
                content, reasoning, fin = _consume_stream(resp)   # có thể raise nếu tunnel rớt giữa stream
                text = (content or reasoning or "").strip()       # ưu tiên content, thiếu -> reasoning
                if text:
                    return _strip_think(text)
                last = (f"LLM trả nội dung RỖNG (finish_reason={fin}). "
                        + (f"Qwen3.5 sinh THINKING dài hơn max_tokens={max_tokens} "
                           "trước khi ra câu trả lời -> TĂNG llm.max_tokens trong config.yaml."
                           if fin == "length"
                           else "Server trả cả content lẫn reasoning_content đều rỗng."))
                break   # deterministic (seed cố định) -> retry vô ích, dừng luôn
            elif status == 402:
                last = ("HF Inference HẾT CREDIT (402). Nạp credit/HF PRO hoặc đổi "
                        "provider sang openai_compat (notebook). " + resp.text[:120])
                break   # hết credit -> retry vô ích
            else:
                # 502/503/504 = gateway; 520-527/530 = Cloudflare tunnel báo origin chết.
                # (Với streaming, 524-timeout-do-sinh-lâu KHÔNG còn xảy ra.)
                server_down = status in (502, 503, 504, 520, 521, 522,
                                         523, 524, 525, 526, 527, 530)
                last = f"HTTP {status}: {resp.text[:200]}"
        except requests.RequestException as e:
            server_down, last = True, str(e)                 # mất kết nối = tunnel/server chết
        except LLMError as e:
            server_down, last = True, str(e)                 # discover URL lỗi (ntfy blip) -> tạm thời

        if server_down:
            _discovered["base"] = None          # buộc discover URL mới ở vòng sau
            if time.time() < down_deadline:
                if not announced and get("llm.provider") == "openai_compat":
                    mins = int((down_deadline - time.time()) / 60) + 1
                    print(f"\n  [CHỜ SERVER] LLM có vẻ đã tắt: {str(last)[:70]}\n"
                          f"  → Đang chờ & tự dò URL mới tới ~{mins} phút. HÃY CHẠY LẠI "
                          f"notebook Kaggle (giữ kernel mở); pipeline TỰ TIẾP TỤC vụ này.",
                          flush=True)
                    announced = True
                time.sleep(poll)
                continue                        # chờ server sống lại (không tính là bỏ cuộc)
        else:
            short_left -= 1                     # lỗi KHÁC (không phải server chết)
            if short_left > 0:
                time.sleep(1.5)
                continue
        break                                   # server chết quá deadline, HOẶC hết retry ngắn

    hint = ""
    if get("llm.provider") == "openai_compat":
        hint = (f"\n  → vLLM server (notebook) vẫn KHÔNG sống lại sau khi chờ "
                f"{get('llm.server_down_retry_seconds',600)}s. Chạy lại notebook "
                f"serving/kaggle-qwen-vllm-server.ipynb (giữ kernel mở) — nó publish URL mới "
                f"lên ntfy '{NTFY_TOPIC}', rồi chạy LẠI cùng --tag để tiếp tục từ vụ dở.")
    raise LLMError(f"LLM server lỗi (đã chờ/thử): {last}{hint}")


def llm(prompt: str, system: str | None = None,
        temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Gọi LLM, trả text. Mọi agent dùng chung. Có cache đĩa."""
    system = system or VI_SYSTEM
    temperature = get("llm.temperature", 0.2) if temperature is None else temperature
    max_tokens = max_tokens or get("llm.max_tokens", 1024)

    if get("llm.cache", True):
        key = _cache_key(prompt, system, temperature, max_tokens)
        cached = _load_cache().get(key)
        if cached is not None:
            return cached

    if get("llm.provider") == "anthropic":
        raise LLMError("provider=anthropic KHÔNG hợp lệ cho cuộc thi (chỉ open-weight <10B).")
    out = _call_hf(prompt, system, temperature, max_tokens)

    if get("llm.cache", True):
        _store(key, out)
    return out


def extract_json(text: str) -> dict:
    """
    Bóc JSON object đầu tiên từ output LLM; chịu được fence ```json và output bị
    CẮT CỤT (tự đóng ngoặc còn thiếu rồi parse lại).
    """
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rsplit("```", 1)[0].strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"Không thấy JSON: {text[:150]}")
    depth, end = 0, None
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    cand = text[start:end] if end else text[start:]
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        rep = cand.rstrip().rstrip(",")
        ob, br = rep.count("{") - rep.count("}"), rep.count("[") - rep.count("]")
        if ob > 0 or br > 0:
            rep += "]" * max(0, br) + "}" * max(0, ob)
            try:
                return json.loads(rep)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"JSON không parse được (có thể cắt cụt): {cand[:150]}")
