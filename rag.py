"""
Retrieval untuk pertanyaan umum (FAQ + kebijakan).
Pakai TF-IDF (ringan, jalan cepat di CPU, ga butuh model embedding gede).
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# kata sapaan yang muncul di HAMPIR SEMUA pertanyaan FAQ (kak, min, admin, dst)
# harus dibuang dari perhitungan TF-IDF, kalau tidak dia jadi kata "penentu" palsu
# dan bikin pertanyaan generik (mis. cuma ketik "cs") match ke jawaban random.
SAPAAN_STOPWORDS = [
    "kak", "min", "admin", "gan", "cs", "ya", "dong", "nih", "kok", "deh",
    "sih", "dan", "yang", "di", "ke", "dari", "untuk", "itu", "ini", "apa",
    "gimana", "bagaimana", "apakah", "atau", "juga", "saja", "kalau", "kalo",
    "permisi", "halo", "hai", "selamat", "pagi", "siang", "sore", "malam",
]


class FaqRetriever:
    def __init__(self, faq_path="data/faq.json", kebijakan_path="data/kebijakan.json"):
        with open(faq_path, encoding="utf-8") as f:
            faq = json.load(f)
        with open(kebijakan_path, encoding="utf-8") as f:
            kebijakan = json.load(f)

        # gabungkan faq + kebijakan jadi satu koleksi dokumen yang bisa dicari
        self.docs = []
        for item in faq:
            self.docs.append({
                "teks_cari": item["pertanyaan"],
                "jawaban": item["jawaban"],
                "kategori": item["kategori"],
                "sumber": "faq",
            })
        for item in kebijakan:
            self.docs.append({
                "teks_cari": item["judul"] + " " + item["isi"],
                "jawaban": item["isi"],
                "kategori": item["kategori"],
                "sumber": "kebijakan",
            })

        corpus = [d["teks_cari"] for d in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words=SAPAAN_STOPWORDS)
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_k=3, min_score=0.25):
        q_vec = self.vectorizer.transform([query])
        # kalau setelah dibuang stopword-nya query jadi kosong (mis. cuma "min" doang),
        # vektornya nol semua -> jangan match apa-apa
        if q_vec.nnz == 0:
            return []

        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked_idx = scores.argsort()[::-1][:top_k]

        results = []
        for i in ranked_idx:
            if scores[i] < min_score:
                continue
            d = self.docs[i]
            results.append({
                "jawaban": d["jawaban"],
                "kategori": d["kategori"],
                "sumber": d["sumber"],
                "score": float(scores[i]),
            })
        return results
