import json
import time
from flask import Flask, render_template, request, jsonify

from rag import FaqRetriever
from produk_search import ProdukSearch
from llm import generate_answer_stream

app = Flask(__name__)

faq_retriever = FaqRetriever()
produk_search = ProdukSearch()

with open("data/toko.json", encoding="utf-8") as f:
    TOKO_INFO = json.load(f)

STOK_HARGA_KEYWORDS = ["harga", "stok", "ada gak", "ada ga", "tersedia", "berapa",
                       "warna apa", "warna apa aja", "sewa"]
TOKO_KEYWORDS = ["alamat", "lokasi", "jam buka", "jam operasional", "kontak", "whatsapp",
                  "wa toko", "rekening", "qris", "instagram", "buka jam"]
TANGGAL_KEYWORDS = ["tanggal", "besok", "lusa", "minggu depan", "bulan depan"]

# "ingatan" sederhana lintas-pesan: kategori/ukuran/warna terakhir yang kesebut,
# biar pesan susulan kayak "ukuran XXL warna navy" (tanpa nyebut ulang "jas") masih nyambung.
# CATATAN: ini state global sederhana, cukup buat demo satu percakapan/satu user.
# Kalau nanti dipakai multi-user beneran, ini perlu dipindah ke session per-user.
LAST_FILTER = {"kategori": None, "ukuran": None, "warna": None}


def build_toko_context():
    rek = "; ".join(f"{r['bank']}: {r['nomor_rekening']} a.n {r['atas_nama']}"
                     for r in TOKO_INFO["rekening_pembayaran"])
    return (f"Nama toko: {TOKO_INFO['nama']}\n"
            f"Alamat: {TOKO_INFO['alamat']}\n"
            f"Jam operasional: {TOKO_INFO['jam_operasional']} ({TOKO_INFO['hari_operasional']})\n"
            f"WhatsApp: {TOKO_INFO['whatsapp']}\n"
            f"Email: {TOKO_INFO['email']}\n"
            f"Instagram: {TOKO_INFO['instagram']}\n"
            f"Rekening pembayaran: {rek}\n"
            f"QRIS tersedia: {TOKO_INFO['qris']['tersedia']}")


import re

GREETING_WORDS = ["hai", "halo", "hi", "permisi", "selamat pagi", "selamat siang",
                   "selamat sore", "selamat malam", "assalamualaikum"]


def _ada_kata_utuh(kata_list, text):
    """Cek kata sebagai kata utuh (word boundary), bukan substring bebas.
    Biar 'hi' ga ke-match ke 'hitam', 'ada' ga ke-match ke 'sedang', dst."""
    for kata in kata_list:
        if re.search(r"\b" + re.escape(kata) + r"\b", text):
            return True
    return False


def build_produk_context(hasil, filter_info):
    if not hasil:
        return (f"Tidak ditemukan produk yang cocok dengan filter "
                f"kategori={filter_info['kategori']}, ukuran={filter_info['ukuran']}, "
                f"warna={filter_info['warna']}.")

    # kalau pelanggan belum sebutin ukuran ATAU warna spesifik, jangan asal comot
    # beberapa produk buat dijawab - kasih ringkasan pilihan yang ada, biar LLM
    # nanya balik ke pelanggan mau pilih yang mana.
    if not filter_info["ukuran"] and not filter_info["warna"]:
        semua_ukuran = sorted(set(p["ukuran"] for p in hasil))
        semua_warna = sorted(set(p["warna"] for p in hasil))
        harga_min = min(p["harga_sewa_per_hari"] for p in hasil)
        harga_max = max(p["harga_sewa_per_hari"] for p in hasil)
        return (
            f"Kategori {filter_info['kategori']} tersedia, TAPI pelanggan belum menyebutkan "
            f"ukuran/warna spesifik. Harga berkisar Rp{harga_min:,} - Rp{harga_max:,} per hari. "
            f"Beberapa pilihan warna tersedia: {', '.join(semua_warna[:6])}. "
            f"Ukuran tersedia: {', '.join(semua_ukuran)}. "
            f"INSTRUKSI: konfirmasi kategori ini tersedia lalu tanyakan ukuran dan warna yang "
            f"diinginkan pelanggan, jangan menyebut harga produk spesifik dulu."
        )

    lines = []
    for p in hasil[:5]:
        lines.append(
            f"- {p['nama_produk']} | Warna: {p['warna']} | Ukuran: {p['ukuran']} | "
            f"Harga sewa/hari: Rp{p['harga_sewa_per_hari']:,} | Deposit: Rp{p['deposit']:,} | "
            f"Stok tersedia: {p['jumlah_stok']} unit | Status: {p['status']}"
        )
    return "\n".join(lines)


def _guide_satu_kategori(jenis):
    if jenis == "sepatu":
        data = produk_search.panduan_ukuran("sepatu")
        return "Panduan ukuran sepatu:\n" + "\n".join(
            f"- Ukuran {d['ukuran']}: panjang kaki {d['panjang_kaki_cm']} cm" for d in data)
    if jenis == "celana":
        data = produk_search.panduan_ukuran("celana")
        return "Panduan ukuran celana:\n" + "\n".join(
            f"- Ukuran {d['ukuran']}: lingkar pinggang {d['lingkar_pinggang_cm']} cm, "
            f"panjang {d['panjang_celana_cm']} cm" for d in data)
    # jas
    data = produk_search.panduan_ukuran("jas")
    return "Panduan ukuran jas:\n" + "\n".join(
        f"- Ukuran {d['ukuran']}: tinggi {d['tinggi_badan_cm']} cm, berat {d['berat_badan_kg']} kg, "
        f"lingkar dada {d['lingkar_dada_cm']} cm" for d in data)


def build_ukuran_context(jenis_list):
    """jenis_list: list berisi kombinasi dari 'jas'/'celana'/'sepatu'."""
    return "\n\n".join(_guide_satu_kategori(j) for j in jenis_list)


def _kategori_disebut_list(text):
    """Cari SEMUA kategori yang disebut di teks (bisa lebih dari satu, mis. 'celana dan sepatunya').
    Toleran imbuhan (sepatunya, celananya, dst), tapi 'jas' sengaja gak nyangkut ke 'jasa'."""
    hasil = []
    if re.search(r"\bjas(?!a)\w*\b", text):
        hasil.append("jas")
    if re.search(r"\bcelana\w*\b", text):
        hasil.append("celana")
    if re.search(r"\b(sepatu|pantofel)\w*\b", text):
        hasil.append("sepatu")
    return hasil


_JENIS_KE_KATEGORI_PRODUK = {"jas": "Jas", "celana": "Celana", "sepatu": "Sepatu Pantofel"}
_KATEGORI_PRODUK_KE_JENIS = {v: k for k, v in _JENIS_KE_KATEGORI_PRODUK.items()}


def route_and_get_context(message):
    text_lower = message.lower().strip()

    # sapaan doang / pesan kependekan -> jangan coba di-retrieve, gampang ngaco
    kata_bersih = re.sub(r"\b(min|kak|cs)\b", "", text_lower).strip()
    if _ada_kata_utuh(GREETING_WORDS, text_lower) or len(kata_bersih) < 3:
        LAST_FILTER["kategori"] = None
        LAST_FILTER["ukuran"] = None
        LAST_FILTER["warna"] = None
        return ("SAPAAN: Pelanggan baru menyapa atau pesannya terlalu singkat/umum untuk "
                "dicari jawabannya. INSTRUKSI: sambut dengan ramah dan tanyakan mau tanya "
                "atau sewa apa (jas/celana/sepatu), jangan mengarang informasi apa pun.")

    # deteksi ukuran/warna/kategori spesifik dari teks - dipakai di beberapa cabang di bawah
    ukuran_baru = produk_search._cari_ukuran(text_lower)
    warna_baru = produk_search._cari_warna(text_lower)
    kategori_baru = produk_search._cari_kategori(text_lower)

    # pertanyaan soal PANDUAN ukuran (bukan cek stok produk spesifik):
    # ciri-cirinya ada kata "ukuran" TAPI gak nyebut nilai ukuran/warna konkret
    # (kalau udah nyebut nilai kayak "L"/"32"/warna, itu berarti nanya produk spesifik -> ke cabang produk)
    if "ukuran" in text_lower and not ukuran_baru and not warna_baru:
        kategori_list = _kategori_disebut_list(text_lower)
        if not kategori_list and LAST_FILTER["kategori"]:
            jenis_terakhir = _KATEGORI_PRODUK_KE_JENIS.get(LAST_FILTER["kategori"])
            if jenis_terakhir:
                kategori_list = [jenis_terakhir]
        if not kategori_list:
            return ("Pelanggan menanyakan panduan ukuran tapi belum jelas untuk kategori apa "
                    "(jas/celana/sepatu). INSTRUKSI: tanyakan balik mau panduan ukuran untuk "
                    "jas, celana, atau sepatu.")
        # simpan kategori yang baru kesebut (kalau cuma satu) biar nyambung ke pesan berikutnya
        if len(kategori_list) == 1:
            LAST_FILTER["kategori"] = _JENIS_KE_KATEGORI_PRODUK[kategori_list[0]]
        return build_ukuran_context(kategori_list)

    if _ada_kata_utuh(TOKO_KEYWORDS, text_lower):
        return build_toko_context()

    ada_kategori = bool(kategori_baru)
    ada_sinyal_produk = _ada_kata_utuh(STOK_HARGA_KEYWORDS, text_lower)
    topik_produk_aktif = ada_kategori or bool(LAST_FILTER["kategori"])

    # cek proteksi tanggal DULUAN, sebelum logic produk lain - kalau lagi ngomongin
    # produk (baru atau lanjutan) dan nyebut tanggal, jangan biarin LLM ngarang ketersediaan
    if topik_produk_aktif and _ada_kata_utuh(TANGGAL_KEYWORDS, text_lower):
        return ("SISTEM BELUM BISA cek ketersediaan berdasarkan tanggal booking secara "
                "otomatis. INSTRUKSI: jangan mengonfirmasi atau menolak ketersediaan "
                "tanggal apa pun, sampaikan jujur bahwa pengecekan tanggal booking perlu "
                "dikonfirmasi langsung ke CS.")

    # pesan lanjutan yang nyebut ukuran/warna tapi gak nyebut ulang kategori
    # (mis. "ukuran XXL warna navy" abis sebelumnya nanya soal jas)
    lanjutan_tanpa_kategori = (not kategori_baru) and (ukuran_baru or warna_baru) and LAST_FILTER["kategori"]

    if ada_kategori and ada_sinyal_produk or lanjutan_tanpa_kategori:
        kategori_final = kategori_baru or LAST_FILTER["kategori"]
        ukuran_final = ukuran_baru or (LAST_FILTER["ukuran"] if not kategori_baru else None)
        warna_final = warna_baru or (LAST_FILTER["warna"] if not kategori_baru else None)

        # simpan buat pesan berikutnya
        LAST_FILTER["kategori"] = kategori_final
        LAST_FILTER["ukuran"] = ukuran_final
        LAST_FILTER["warna"] = warna_final

        hasil = produk_search.cari_produk_by_filter(
            kategori_final, ukuran_final, warna_final, max_hasil=50)
        filter_info = {"kategori": kategori_final, "ukuran": ukuran_final, "warna": warna_final}
        return build_produk_context(hasil, filter_info)

    # fallback: FAQ + kebijakan
    hasil = faq_retriever.search(message, top_k=3)
    if not hasil:
        return "Tidak ditemukan informasi yang relevan di data FAQ maupun kebijakan toko."
    return "\n".join(f"- ({h['kategori']}) {h['jawaban']}" for h in hasil)


@app.route("/")
def index():
    return render_template("index.html", nama_toko=TOKO_INFO["nama"])


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = (data or {}).get("message", "").strip()
    if not message:
        return jsonify({"reply": "Pesannya kosong nih, coba ketik pertanyaannya ya."})

    t0 = time.time()
    context_text = route_and_get_context(message)
    print(f"[TIMING] cari konteks: {time.time() - t0:.2f}s")

    def generate():
        t1 = time.time()
        for piece in generate_answer_stream(message, context_text):
            yield piece
        print(f"[TIMING] panggil Ollama (streaming selesai): {time.time() - t1:.2f}s")

    return app.response_class(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5000)