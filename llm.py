"""
Pemanggil Ollama API. Pastikan Ollama sudah jalan (`ollama serve`)
dan model sudah di-pull (`ollama pull qwen2.5:3b`).
"""
import os
import json
import multiprocessing
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "qwen2.5:1.5b")
CPU_THREADS = multiprocessing.cpu_count()

SYSTEM_PROMPT = """Kamu admin CS toko Seven Inc, rental jas/celana/sepatu pantofel di Yogyakarta. \
Jawab ramah, singkat (maks 3 kalimat), gaya CS toko online.
ATURAN:
- Jawab HANYA dalam Bahasa Indonesia. Jangan pernah pakai bahasa lain sama sekali.
- Jawab HANYA berdasarkan INFORMASI di bawah. Kalau ada angka/harga yang TIDAK tertulis \
eksplisit di INFORMASI (termasuk harga paket gabungan), JANGAN sebut angka apa pun - \
bilang perlu dihitungkan CS dan arahkan hubungi CS langsung.
- Kalau info kurang, jujur bilang. Jangan sebut kamu AI."""


def _build_payload(user_message, context_text, stream):
    prompt = f"""INFORMASI YANG TERSEDIA:
{context_text}

PERTANYAAN PELANGGAN:
{user_message}

Jawab pertanyaan pelanggan di atas berdasarkan informasi yang tersedia."""

    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
        "keep_alive": "30m",   # model tetap di RAM 30 menit, gak perlu reload tiap request
        "options": {
            "num_predict": 150,
            "num_ctx": 2048,
            "num_thread": CPU_THREADS,
            "temperature": 0.1,
            "repeat_penalty": 1.15,
        }
    }


def generate_answer(user_message, context_text):
    """Versi non-streaming (dipakai kalau butuh jawaban utuh sekaligus)."""
    payload = _build_payload(user_message, context_text, stream=False)
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return ("Maaf, sistem chatbot sedang tidak bisa terhubung ke server AI lokal. "
                "Pastikan Ollama sudah jalan (`ollama serve`) dan model sudah di-pull.")
    except Exception as e:
        return f"Terjadi kesalahan saat memproses jawaban: {e}"


def generate_answer_stream(user_message, context_text):
    """Versi streaming - yield potongan teks begitu dihasilkan Ollama."""
    payload = _build_payload(user_message, context_text, stream=True)
    try:
        with requests.post(OLLAMA_URL, json=payload, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if chunk.get("done"):
                    break
    except requests.exceptions.ConnectionError:
        yield ("Maaf, sistem chatbot sedang tidak bisa terhubung ke server AI lokal. "
               "Pastikan Ollama sudah jalan (`ollama serve`) dan model sudah di-pull.")
    except Exception as e:
        yield f"Terjadi kesalahan saat memproses jawaban: {e}"