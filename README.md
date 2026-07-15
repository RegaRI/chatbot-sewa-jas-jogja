# Chatbot Seven Inc

## Cara jalanin
1. Install Ollama (kalau belum): https://ollama.com
2. Pull model: `ollama pull qwen2.5:3b`
3. Pastikan Ollama jalan: `ollama serve` (biasanya otomatis jalan setelah install)
4. Install dependency Python:
   ```
   pip install -r requirements.txt
   ```
5. Jalankan:
   ```
   python app.py
   ```
6. Buka browser: http://localhost:5000

## Struktur
- `data/` — semua file JSON (faq, kebijakan, produk, stok, toko, ukuran)
- `rag.py` — pencarian FAQ & kebijakan pakai TF-IDF (buat pertanyaan umum)
- `produk_search.py` — pencarian data terstruktur (harga, stok, ukuran) — TIDAK lewat LLM biar angkanya akurat
- `llm.py` — pemanggil Ollama API, nyusun jawaban natural dari data yang ditemukan
- `app.py` — Flask app, routing pertanyaan ke modul yang sesuai
- `templates/index.html` + `static/style.css` — tampilan chat

## Cara kerja singkat
1. User kirim pesan
2. `app.py` cek isi pesan: nanya soal ukuran badan → panduan ukuran, nanya alamat/jam/kontak → info toko, nanya harga/stok produk spesifik → cari di produk.json + stok.json, selain itu → cari di FAQ/kebijakan pakai TF-IDF
3. Hasil pencarian (bukan jawaban akhir) dikirim ke Qwen sebagai konteks
4. Qwen susun jawaban natural gaya CS toko berdasarkan konteks itu

## Kalau mau ganti model
Set environment variable sebelum jalanin:
```
set OLLAMA_MODEL=qwen2.5:1.5b        (Windows)
```

## Tuning lanjutan (opsional, kalau hasilnya kurang pas)
- Kalau retrieval FAQ sering meleset, coba turunkan `min_score` di `rag.py` (default 0.15)
- Kalau deteksi kategori/ukuran/warna di `produk_search.py` sering salah tangkap, itu bagian paling gampang buat di-debug — tinggal print `filter_info` di `app.py` buat lihat apa yang kedetect
