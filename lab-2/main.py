import os
import re
import json
import string
from collections import Counter

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize, wordpunct_tokenize, TreebankWordTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag

import spacy
from spacy.lang.en import English
from langdetect import detect, LangDetectException


OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

def is_english(text):
    try:
        return detect(text[:500]) == "en"
    except LangDetectException:
        return False

def save_text(filename, text):
    with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [+] збережено: {os.path.join(OUT_DIR, filename)}")


def scrape_jobs(query="nlp+engineer"):
    texts = []
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    NLP_KW = {"nlp", "natural language", "machine learning",
              "language model", "text mining", "deep learning", "data science"}

    try:
        r = requests.get("https://remoteok.com/api", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                if not isinstance(item, dict):
                    continue
                title = (item.get("position", "") or "").lower()
                desc  = item.get("description", "") or ""
                if any(kw in title or kw in desc.lower() for kw in NLP_KW):
                    clean = BeautifulSoup(desc, "html.parser").get_text(" ")
                    if len(clean) > 100 and is_english(clean):
                        texts.append(clean)
            if texts:
                print(f"  [+] RemoteOK: отримано {len(texts)} вакансій")
        else:
            print(f"  [!] RemoteOK: HTTP {r.status_code} — {r.text[:60].strip()}")
    except Exception as e:
        print(f"  [!] RemoteOK: {e.__class__.__name__}: {e}")

    if len(texts) < 3:
        try:
            r = requests.get(
                f"https://www.arbeitnow.com/api/job-board-api?search={query}",
                headers=HEADERS, timeout=10
            )
            if r.status_code == 200:
                for item in r.json().get("data", []):
                    desc = item.get("description", "") or ""
                    clean = BeautifulSoup(desc, "html.parser").get_text(" ")
                    if len(clean) > 100 and is_english(clean):
                        texts.append(clean)
                print(f"  [+] Arbeitnow: всього зібрано {len(texts)} вакансій")
            else:
                print(f"  [!] Arbeitnow: HTTP {r.status_code} — {r.text[:60].strip()}")
        except Exception as e:
            print(f"  [!] Arbeitnow: {e.__class__.__name__}: {e}")

    if len(texts) < 3:
        try:
            r = requests.get(
                f"https://himalayas.app/jobs/api?q={query}&limit=20",
                headers=HEADERS, timeout=10
            )
            if r.status_code == 200:
                for item in r.json().get("jobs", []):
                    desc = item.get("description", "") or ""
                    clean = BeautifulSoup(desc, "html.parser").get_text(" ")
                    if len(clean) > 100 and is_english(clean):
                        texts.append(clean)
                print(f"  [+] Himalayas: всього зібрано {len(texts)} вакансій")
            else:
                print(f"  [!] Himalayas: HTTP {r.status_code} — {r.text[:60].strip()}")
        except Exception as e:
            print(f"  [!] Himalayas: {e.__class__.__name__}: {e}")

    if not texts:
        print("  [!] Жодне джерело не відповіло. Перевір інтернет-з'єднання.")

    return texts


def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation.replace("+", "").replace("#", "")))
    return text


def tokenize_three_ways(text):
    tok_word = word_tokenize(text)

    tok_punct = wordpunct_tokenize(text)

    nlp_blank = English()
    doc = nlp_blank(text)
    tok_spacy = [t.text for t in doc if not t.is_space]

    return tok_word, tok_punct, tok_spacy


def remove_stopwords(tokens):
    sw = set(stopwords.words("english"))
    sw |= {"experience", "skills", "work", "team", "require", "required",
           "requirements", "responsibilities", "ability", "strong", "good",
           "plus", "field", "related", "year", "years", "join", "looking",
           "candidate", "knowledge", "understanding", "familiar", "familiarity",
           "preferred", "degree", "us", "build", "building", "develop",
           "developing", "working", "etc", "must", "well", "new"}
    return [t for t in tokens if t not in sw and len(t) > 1]


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in tagged]


def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]


def top_keywords(tokens, n=10):
    return Counter(tokens).most_common(n)


SKILL_TERMS = {
    "python": ["python"],
    "pytorch": ["pytorch"],
    "tensorflow": ["tensorflow"],
    "transformers": ["transformer", "transformers"],
    "bert": ["bert"],
    "gpt / llm": ["gpt", "llm", "large language model", "language model"],
    "hugging face": ["hugging", "huggingface"],
    "deep learning": ["deep learning", "neural network", "neural"],
    "spacy / nltk": ["spacy", "nltk"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "cloud (aws/gcp/azure)": ["aws", "gcp", "azure", "cloud"],
    "mlops / ci-cd": ["mlops", "ci/cd", "cicd", "pipeline", "pipelines"],
    "sql / data": ["sql", "pandas", "numpy", "scikit", "spark", "etl"],
    "embeddings / vector db": ["embedding", "embeddings", "vector"],
    "communication": ["communication", "collaboration", "teamwork"],
}


def analyze_skills(text_lower):
    counts = {}
    for skill, variants in SKILL_TERMS.items():
        c = sum(text_lower.count(v) for v in variants)
        if c > 0:
            counts[skill] = c
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


def plot_top(counter_list, title, filename):
    words = [w for w, _ in counter_list]
    freqs = [c for _, c in counter_list]
    plt.figure(figsize=(10, 5))
    plt.barh(words[::-1], freqs[::-1], color="#3b6fb5")
    plt.xlabel("Частота")
    plt.title(title)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [+] графік збережено: {path}")


if __name__ == "__main__":
    print("=" * 70)
    print(" Домінантні вимоги до посади NLP Engineer")
    print("=" * 70)

    print("\n[1] WEB-СКРАПІНГ ВАКАНСІЙ")
    raw_jobs = scrape_jobs("nlp engineer")
    raw_corpus = "\n\n".join(raw_jobs)
    if not raw_corpus.strip():
        print("  [!] Корпус порожній — нема що аналізувати. Перевір з'єднання.")
        exit(1)
    save_text("1_raw_input.txt", raw_corpus)
    print(f"  Зібрано описів вакансій: {len(raw_jobs)}")

    print("\n[2] ФІЛЬТРАЦІЯ")
    filtered = clean_text(raw_corpus)
    save_text("2_filtered.txt", filtered)
    print(f"  Довжина після фільтрації: {len(filtered)} символів")

    print("\n[3] НОРМАЛІЗАЦІЯ")
    normalized = normalize_text(filtered)
    save_text("3_normalized.txt", normalized)
    print(f"  Приклад: {normalized[:120]}...")

    print("\n[4] ТОКЕНІЗАЦІЯ (3 типи)")
    tw, tp, ts = tokenize_three_ways(normalized)
    save_text("4_tokens_word.txt", " ".join(tw))
    save_text("4_tokens_wordpunct.txt", " ".join(tp))
    save_text("4_tokens_spacy.txt", " ".join(ts))
    print(f"  word_tokenize:      {len(tw)} токенів")
    print(f"  wordpunct_tokenize: {len(tp)} токенів")
    print(f"  spaCy tokenizer:    {len(ts)} токенів")

    tokens = tw

    print("\n[5] ВИДАЛЕННЯ СТОП-СЛІВ")
    no_sw = remove_stopwords(tokens)
    save_text("5_no_stopwords.txt", " ".join(no_sw))
    print(f"  Було токенів: {len(tokens)}, стало: {len(no_sw)}")

    print("\n[6] ЛЕМАТИЗАЦІЯ")
    lemmas = lemmatize_tokens(no_sw)
    save_text("6_lemmas.txt", " ".join(lemmas))
    print(f"  Приклад лем: {lemmas[:15]}")

    print("\n[7] СТЕМІНГ")
    stems = stem_tokens(no_sw)
    save_text("7_stems.txt", " ".join(stems))
    print(f"  Приклад стемів: {stems[:15]}")

    print("\n[8] ТОП-10 ДОМІНАНТНИХ ВИМОГ")
    top_lemmas = top_keywords(lemmas, 10)
    print("  За лемами:")
    for i, (w, c) in enumerate(top_lemmas, 1):
        print(f"    {i:>2}. {w:<18} {c}")

    df = pd.DataFrame(top_lemmas, columns=["keyword", "frequency"])
    df.to_csv(os.path.join(OUT_DIR, "8_top_keywords.csv"),
              index=False, encoding="utf-8")
    print(f"  [+] CSV збережено: {os.path.join(OUT_DIR, '8_top_keywords.csv')}")

    plot_top(top_lemmas, "Топ-10 домінантних вимог (NLP Engineer)",
             "8_top_keywords.png")
    top_stems = top_keywords(stems, 10)
    plot_top(top_stems, "Топ-10 за основами слів (стемінг)",
             "8_top_stems.png")

    print("\n[9] ДОМІНАНТНІ ТЕХНІЧНІ ВИМОГИ (за словником навичок)")
    skills = analyze_skills(normalized)
    for i, (s, c) in enumerate(skills, 1):
        print(f"    {i:>2}. {s:<24} {c}")
    df_sk = pd.DataFrame(skills, columns=["skill", "mentions"])
    df_sk.to_csv(os.path.join(OUT_DIR, "9_top_skills.csv"),
                 index=False, encoding="utf-8")
    plot_top(skills[:12], "Домінантні вимоги до посади NLP Engineer",
             "9_top_skills.png")

    summary = {
        "jobs_analyzed": len(raw_jobs),
        "tokens_word": len(tw),
        "tokens_wordpunct": len(tp),
        "tokens_spacy": len(ts),
        "tokens_after_stopwords": len(no_sw),
        "top_keywords": dict(top_lemmas),
        "top_skills": dict(skills),
    }
    save_text("9_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))

    print("\n" + "=" * 70)
    print(" Аналіз завершено. Усі результати — у каталозі ./output/")
    print("=" * 70)
