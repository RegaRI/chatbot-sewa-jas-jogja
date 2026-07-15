"""
Pencarian data terstruktur (produk, stok, ukuran).
Ini SENGAJA tidak lewat LLM/RAG karena datanya presisi (harga, deposit, jumlah stok)
- kalau diserahkan ke LLM buat 'mengingat' angka, gampang halusinasi.
"""
import json


class ProdukSearch:
    def __init__(self, produk_path="data/produk.json", stok_path="data/stok.json",
                 ukuran_path="data/ukuran.json"):
        with open(produk_path, encoding="utf-8") as f:
            self.produk = json.load(f)["produk"]
        with open(stok_path, encoding="utf-8") as f:
            stok_list = json.load(f)
        with open(ukuran_path, encoding="utf-8") as f:
            self.ukuran_guide = json.load(f)

        # index stok by (id_produk, ukuran, warna) biar cepat dicari
        self.stok_index = {}
        for s in stok_list:
            key = (s["id_produk"], s["ukuran"], s["warna"])
            self.stok_index[key] = s

        self.kategori_map = {
            "jas": "Jas",
            "celana": "Celana",
            "sepatu": "Sepatu Pantofel",
            "pantofel": "Sepatu Pantofel",
        }

    def _cari_kategori(self, text):
        text = text.lower()
        for keyword, kategori in self.kategori_map.items():
            if keyword in text:
                return kategori
        return None

    def _cari_ukuran(self, text):
        text_upper = text.upper()
        kemungkinan = ["5XL", "4XL", "3XL", "XXL", "XL", "XS", "S", "M", "L"]
        for u in kemungkinan:
            # cari ukuran sebagai kata terpisah biar 'S' ga match ke 'SEWA'
            if f" {u} " in f" {text_upper} " or text_upper == u:
                return u
        # ukuran angka (celana/sepatu)
        import re
        angka = re.findall(r"\b\d{2}\b", text)
        return angka[0] if angka else None

    def _cari_warna(self, text):
        text_lower = text.lower()
        semua_warna = {p["warna"].lower(): p["warna"] for p in self.produk}
        for w_lower, w_asli in semua_warna.items():
            if w_lower in text_lower:
                return w_asli
        return None

    def cari_produk_by_filter(self, kategori=None, ukuran=None, warna=None, max_hasil=5):
        """Sama kayak cari_produk, tapi filter dikasih langsung (bukan diparsing dari teks).
        Dipakai buat gabungin info dari pesan sebelumnya (mis. kategori udah disebut duluan)."""
        hasil = []
        for p in self.produk:
            if kategori and p["kategori"] != kategori:
                continue
            if ukuran and p["ukuran"] != ukuran:
                continue
            if warna and p["warna"] != warna:
                continue
            hasil.append(p)
            if len(hasil) >= max_hasil:
                break

        for p in hasil:
            key = (p["id_produk"], p["ukuran"], p["warna"])
            s = self.stok_index.get(key)
            p["jumlah_stok"] = s["jumlah_stok"] if s else 0

        return hasil

    def cari_produk(self, text, max_hasil=5):
        """Cari produk berdasar kata kunci bebas (kategori/warna/ukuran disebut di teks)."""
        kategori = self._cari_kategori(text)
        ukuran = self._cari_ukuran(text)
        warna = self._cari_warna(text)

        hasil = []
        for p in self.produk:
            if kategori and p["kategori"] != kategori:
                continue
            if ukuran and p["ukuran"] != ukuran:
                continue
            if warna and p["warna"] != warna:
                continue
            hasil.append(p)
            if len(hasil) >= max_hasil:
                break

        # tempelkan info stok real-time ke tiap hasil
        for p in hasil:
            key = (p["id_produk"], p["ukuran"], p["warna"])
            s = self.stok_index.get(key)
            p["jumlah_stok"] = s["jumlah_stok"] if s else 0

        return hasil, {"kategori": kategori, "ukuran": ukuran, "warna": warna}

    def panduan_ukuran(self, jenis):
        """jenis: 'jas' / 'celana' / 'sepatu'"""
        return self.ukuran_guide.get(jenis, [])