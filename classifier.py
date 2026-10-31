"""Klasifikasi sentimen BRImo (LLM murni) untuk Modul 2A dan 2B.

Tangani error JSON sementara, tanpa mengubah label atau mengganti hasilnya
menjadi prediksi buatan. Jika tetap gagal, kembalikan 'invalid'.
"""
import json
import re
import time
from pathlib import Path
from groq import BadRequestError, RateLimitError

LABELS = ["Negatif", "Positif"]
NAMA_FITUR = ["ulasan"]


def load_config(path="config.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _is_json_validation_error(exc):
    """Hanya 400 dengan kode validasi JSON yang layak dicoba ulang."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        e = body.get("error", body)
        if isinstance(e, dict) and e.get("code") == "json_validate_failed":
            return True
    return "json_validate_failed" in str(exc)


def _rate_limit_delay(exc, attempt):
    delay = 10 * (attempt + 1)
    response = getattr(exc, "response", None)
    if response is not None:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(180.0, max(delay, float(value) + 2.0))
            except (TypeError, ValueError):
                pass
    match = re.search(r"(?:try again in|retry after)\s+(\d+(?:\.\d+)?)\s*s", str(exc), re.I)
    if match:
        delay = max(delay, float(match.group(1)) + 2)
    return min(180.0, delay)


def _request(client, cfg, messages, response_format, max_tokens, max_retry):
    for attempt in range(max_retry):
        try:
            return client.chat.completions.create(
                model=cfg["model"],
                temperature=cfg["temperature"],
                max_tokens=max_tokens,
                reasoning_effort=cfg.get("reasoning_effort", "low"),
                messages=messages,
                response_format=response_format,
            )
        except RateLimitError as exc:
            if attempt == max_retry - 1:
                raise  # 429 yang berkepanjangan tidak disamarkan sebagai kesalahan klasifikasi
            delay = _rate_limit_delay(exc, attempt)
            print(f"\nAPI sedang dibatasi. Menunggu {delay:.0f} detik...", flush=True)
            time.sleep(delay)
    raise RuntimeError("Percobaan API habis")


def classify(client, cfg, data, max_retry=3):
    ulasan = str(data.get("ulasan", "")).strip()
    if not ulasan:
        return "invalid"

    labels = cfg.get("labels", LABELS)
    messages = [{"role": "system", "content": cfg["system_prompt"]}]
    for contoh in cfg.get("few_shot", []):
        messages.append({"role": "user", "content": str(contoh["data"]["ulasan"])})
        messages.append({"role": "assistant", "content": json.dumps({"label": contoh["label"]}, ensure_ascii=False)})
    messages.append({"role": "user", "content": ulasan})

    schema = {
        "type": "object",
        "properties": {"label": {"type": "string", "enum": labels}},
        "required": ["label"],
        "additionalProperties": False,
    }
    structured = {"type": "json_schema", "json_schema": {
        "name": "sentimen", "strict": True, "schema": schema
    }}
    fallback = {"type": "json_object"}
    budget = int(cfg.get("max_tokens", 384))
    # Mencoba skema ketat lebih dahulu; fallback hanya jika format API gagal.
    attempts = [
        ("JSON Schema", structured, max(256, budget)),
        ("JSON Schema ulang", structured, max(512, budget)),
        ("JSON Object", fallback, max(512, budget)),
    ]
    for nama_mode, fmt, token_limit in attempts:
        try:
            resp = _request(client, cfg, messages, fmt, token_limit, max_retry)
        except BadRequestError as exc:
            if not _is_json_validation_error(exc):
                raise  # Error konfigurasi/key/schema yang berbeda harus terlihat
            print(f"\nRespons {nama_mode} gagal validasi JSON. Mencoba format lain...", flush=True)
            continue

        content = resp.choices[0].message.content if resp.choices else None
        try:
            obj = json.loads(content or "")
        except (json.JSONDecodeError, TypeError):
            print(f"\nRespons {nama_mode} tidak berisi JSON valid. Mencoba lagi...", flush=True)
            continue
        if isinstance(obj, dict) and obj.get("label") in labels:
            return obj["label"]
        print(f"\nLabel tidak valid dari {nama_mode}. Mencoba lagi...", flush=True)

    print("\nGagal menghasilkan label sah setelah 3 percobaan; dicatat sebagai invalid.", flush=True)
    return "invalid"
