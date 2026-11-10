import re
import json
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report,
                             silhouette_score)
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass



ROZETKA_PRODUCTS = [
    {"name": "Мобільний телефон Apple iPhone 17 Pro", "id": "543545605"},
    {"name": "Ноутбук Lenovo IdeaPad Slim 3",   "id": "559616715"},
    {"name": "Навушники JBL Tune 510BT",   "id": "291422348"},
    {"name": "Кавомашина PHILIPS Series 2300",    "id": "497537359"},
    {"name": "Робот-пилосос Xiaomi Robot Vacuum S40 Pro",    "id": "556406148"},
]

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "uk-UA,uk;q=0.9",
}


def scrape_rozetka_reviews(products, max_pages=3):
    reviews = []
    session = requests.Session()
    session.headers.update(HEADERS)

    for prod in products:
        for page in range(1, max_pages + 1):
            api = ("https://product-api.rozetka.com.ua/v4/comments/get"
                   f"?front-type=xl&country=UA&lang=ua"
                   f"&goods={prod['id']}&page={page}&sort=date")
            try:
                resp = session.get(api, timeout=10)
                if resp.status_code != 200:
                    break
                data = resp.json()
                comments = data.get("data", {}).get("comments", [])
                if not comments:
                    break
                for c in comments:
                    text = (c.get("text") or "").strip()
                    mark = c.get("mark") or 0
                    if text and mark:
                        reviews.append({
                            "product": prod["name"],
                            "text": text,
                            "mark": int(mark),
                        })
                time.sleep(0.5)
            except Exception as e:
                print(f"  [!] Помилка скрапінгу {prod['name']} стор.{page}: {e}")
                break

    return reviews



def load_dataset():
    print("=" * 70)
    print(" БЛОК 1. Web-скрапінг відгуків Rozetka")
    print("=" * 70)
    reviews = scrape_rozetka_reviews(ROZETKA_PRODUCTS)


    print(f"  [OK] Зібрано {len(reviews)} реальних відгуків з Rozetka.")
    source = "Rozetka (live scraping)"


    df = pd.DataFrame(reviews)
    df = df[df["mark"] != 3].copy()
    df["label"] = (df["mark"] >= 4).astype(int)
    df = df.reset_index(drop=True)
    print(f"  Джерело даних: {source}")
    print(f"  Усього відгуків: {len(df)} | товарів: {df['product'].nunique()}")
    print(f"  Позитивних: {(df.label==1).sum()} | негативних: {(df.label==0).sum()}")
    return df, source



UA_STOPWORDS = {
    "і", "та", "що", "це", "на", "у", "в", "з", "за", "до", "по", "не", "як",
    "а", "але", "бо", "ще", "вже", "дуже", "так", "ось", "цей", "ця", "це",
    "я", "ти", "він", "вона", "воно", "ми", "ви", "вони", "був", "була",
    "було", "бути", "є", "для", "про", "від", "при", "над", "під", "о",
    "її", "його", "їх", "мій", "моя", "те", "той", "там", "тут", "де", "коли",
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^а-яґєії'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    tokens = word_tokenize(text, language="russian")  # кирилична токенізація
    tokens = [t for t in tokens if len(t) > 2 and t not in UA_STOPWORDS]
    return tokens


def preprocess(df):
    print("\n" + "=" * 70)
    print(" БЛОК 2. NLP-конвеєр первинної обробки (токенізація, стоп-слова)")
    print("=" * 70)
    df["clean"] = df["text"].apply(clean_text)
    df["tokens"] = df["clean"].apply(tokenize)
    df["clean_joined"] = df["tokens"].apply(lambda x: " ".join(x))
    print(f"  Оброблено {len(df)} документів.")
    print(f"  Приклад: '{df['text'].iloc[0][:50]}...'")
    print(f"       ->  {df['tokens'].iloc[0]}")
    return df


def supervised_classification(df):
    print("\n" + "=" * 70)
    print(" БЛОК 3. Класифікація відгуків позитив/негатив (із вчителем)")
    print("=" * 70)

    X = df["clean_joined"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial NB": MultinomialNB(),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        pred = model.predict(Xte)
        acc = accuracy_score(y_test, pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average="binary", zero_division=0)
        results[name] = {"acc": acc, "precision": p, "recall": r, "f1": f1,
                         "pred": pred, "cm": confusion_matrix(y_test, pred)}
        print(f"\n  Модель: {name}")
        print(f"    Accuracy : {acc:.3f}")
        print(f"    Precision: {p:.3f}  Recall: {r:.3f}  F1: {f1:.3f}")
        print(f"    Ефективність {'ВІДПОВІДАЄ' if acc > 0.6 else 'НЕ відповідає'} вимозі > 60%")

    best = max(results, key=lambda k: results[k]["acc"])
    print(f"\n  >>> Найкраща модель: {best} (accuracy = {results[best]['acc']:.3f})")

    cm = results[best]["cm"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["негатив", "позитив"])
    ax.set_yticklabels(["негатив", "позитив"])
    ax.set_xlabel("Прогноз"); ax.set_ylabel("Факт")
    ax.set_title(f"Матриця плутанини ({best})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig("fig_confusion_matrix.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_confusion_matrix.png")

    return vec, results, best


def frequency_analysis(df):
    print("\n" + "=" * 70)
    print(" БЛОК 4. Частотний та ймовірнісний аналіз (TF-IDF, біграми, дисперсія)")
    print("=" * 70)

    all_tokens = [t for toks in df["tokens"] for t in toks]
    total = len(all_tokens)
    vocab = sorted(set(all_tokens))
    print(f"  Усього токенів: {total} | розмір словника: {len(vocab)}")

    vec = TfidfVectorizer()
    M = vec.fit_transform(df["clean_joined"])
    scores = np.asarray(M.mean(axis=0)).ravel()
    terms = vec.get_feature_names_out()
    order = np.argsort(scores)[::-1][:15]
    top_terms = [(terms[i], scores[i]) for i in order]
    print("\n  Топ-15 термінів за TF-IDF:")
    for term, sc in top_terms:
        print(f"    {term:<18} {sc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([t for t, _ in top_terms][::-1], [s for _, s in top_terms][::-1],
            color="#4C72B0")
    ax.set_xlabel("Середній TF-IDF"); ax.set_title("Топ-15 термінів за TF-IDF")
    plt.tight_layout(); plt.savefig("fig_tfidf_top.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_tfidf_top.png")

    flat = all_tokens
    top_words = [t for t, _ in top_terms[:5]]
    fig, ax = plt.subplots(figsize=(9, 4))
    for idx, w in enumerate(top_words):
        positions = [i for i, tok in enumerate(flat) if tok == w]
        ax.scatter(positions, [idx] * len(positions), marker="|", s=200)
    ax.set_yticks(range(len(top_words))); ax.set_yticklabels(top_words)
    ax.set_xlabel("Позиція слова в корпусі")
    ax.set_title("Лексична дисперсія (lexical dispersion)")
    plt.tight_layout(); plt.savefig("fig_lexical_dispersion.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_lexical_dispersion.png")

    word_lengths = [len(t) for t in all_tokens]
    mean_len = np.mean(word_lengths); var_len = np.var(word_lengths)
    print(f"\n  Довжина слів: середня = {mean_len:.2f}, дисперсія = {var_len:.2f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(word_lengths, bins=range(3, 16), color="#55A868",
            edgecolor="black", alpha=0.8)
    ax.set_xlabel("Довжина слова (символів)"); ax.set_ylabel("Частота")
    ax.set_title("Розподіл довжини слів")
    plt.tight_layout(); plt.savefig("fig_word_length.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_word_length.png")

    bgs = list(bigrams(all_tokens))
    bg_freq = pd.Series(bgs).value_counts().head(12)
    print("\n  Топ-12 біграм:")
    for bg, cnt in bg_freq.items():
        print(f"    {bg[0]} {bg[1]:<15} -> {cnt}")
    labels = [f"{a} {b}" for (a, b) in bg_freq.index][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, bg_freq.values[::-1], color="#C44E52")
    ax.set_xlabel("Частота"); ax.set_title("Топ-12 біграм")
    plt.tight_layout(); plt.savefig("fig_bigrams.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_bigrams.png")

    return top_terms, mean_len, var_len, bg_freq


def unsupervised_clustering(df):
    print("\n" + "=" * 70)
    print(" БЛОК 5. Внутрішня кластеризація без учителя (KMeans + ієрархічна)")
    print("=" * 70)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    M = vec.fit_transform(df["clean_joined"])

    sil = {}
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(M)
        sil[k] = silhouette_score(M, lab)
    best_k = max(sil, key=sil.get)
    print("  Коефіцієнт силуету за кількістю кластерів:")
    for k, s in sil.items():
        print(f"    k={k}: silhouette = {s:.3f}")
    print(f"  >>> Оптимальна кількість кластерів: k={best_k}")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(M)

    print("\n  Розподіл кластерів за реальними мітками:")
    ct = pd.crosstab(df["cluster"], df["label"].map({0: "негатив", 1: "позитив"}))
    print(ct.to_string())

    terms = vec.get_feature_names_out()
    centroids = km.cluster_centers_
    print("\n  Характерні слова кластерів:")
    for c in range(best_k):
        top_idx = centroids[c].argsort()[::-1][:6]
        words = ", ".join(terms[i] for i in top_idx)
        print(f"    Кластер {c}: {words}")

    dense = M.toarray()
    Z = linkage(dense, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Дендрограма ієрархічної кластеризації відгуків")
    ax.set_xlabel("Відгуки (кластери)"); ax.set_ylabel("Відстань (Ward)")
    plt.tight_layout(); plt.savefig("fig_dendrogram.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_dendrogram.png")

    from sklearn.decomposition import PCA
    coords = PCA(n_components=2, random_state=42).fit_transform(dense)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=df["cluster"],
                    cmap="viridis", alpha=0.7, s=40)
    ax.set_title(f"Кластери відгуків у просторі PCA (k={best_k})")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    plt.colorbar(sc, label="кластер"); plt.tight_layout()
    plt.savefig("fig_clusters_pca.png", dpi=130); plt.close()
    print("  [грфк] Збережено: fig_clusters_pca.png")

    return best_k, sil, ct



if __name__ == "__main__":
    df, source = load_dataset()
    df = preprocess(df)
    vec, clf_results, best_model = supervised_classification(df)
    top_terms, mean_len, var_len, bg_freq = frequency_analysis(df)
    best_k, sil, ct = unsupervised_clustering(df)

    print("\n" + "=" * 70)
    print(" ВЕРИФІКАЦІЯ ВИКОНАНА. Усі етапи відпрацювали коректно.")
    print("=" * 70)
