import os
import re
import json
import time
import random

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
_NLP_SPACY = spacy.load("uk_core_news_sm")

import pymorphy3
_MORPH = pymorphy3.MorphAnalyzer(lang='uk')


OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
}


def fetch_html(url, pause=(1.0, 2.5)):
    try:
        time.sleep(random.uniform(*pause))
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.encoding = resp.apparent_encoding
        if resp.status_code == 200:
            return resp.text
        print(f"  [!] {url} -> HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"  [!] помилка запиту {url}: {e}")
    return None


def parse_olx(category_url, category, max_items=40):
    print(f"[OLX] категорія '{category}' -> {category_url}")
    html = fetch_html(category_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items = []
    cards = soup.select('[data-cy="l-card"]')
    for card in cards[:max_items]:
        title_el = card.select_one("h4, h6")
        price_el = card.select_one('[data-testid="ad-price"]')
        location_el = card.select_one('[data-testid="location-date"]')
        title = title_el.get_text(strip=True) if title_el else ""
        price = _extract_number(price_el.get_text() if price_el else "")
        area = _extract_area(title)
        items.append({
            "platform": "OLX",
            "category": category,
            "title": title,
            "price_usd": price,
            "area_m2": area,
            "location": location_el.get_text(strip=True) if location_el else "",
        })
    print(f"  знайдено карток: {len(items)}")
    return items


def parse_domria(category_url, category, max_items=40):
    print(f"[DOM.RIA] категорія '{category}' -> {category_url}")
    html = fetch_html(category_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items = []
    cards = soup.select("section.ticket-clear, div.realty-preview")
    for card in cards[:max_items]:
        title_el = card.select_one("a.blue, .realty-preview-title")
        price_el = card.select_one(".green, .price")
        title = title_el.get_text(strip=True) if title_el else ""
        price = _extract_number(price_el.get_text() if price_el else "")
        area = _extract_area(card.get_text())
        items.append({
            "platform": "DOM.RIA",
            "category": category,
            "title": title,
            "price_usd": price,
            "area_m2": area,
            "location": "Київ",
        })
    print(f"  знайдено карток: {len(items)}")
    return items


def parse_lun(category_url, category, max_items=40):
    print(f"[LUN] категорія '{category}' -> {category_url}")
    html = fetch_html(category_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    items = []
    cards = soup.select('[class*="card"], article')
    for card in cards[:max_items]:
        title_el = card.select_one("h3, h2, a")
        price_el = card.find(string=re.compile(r"\$|грн|\d{3,}"))
        title = title_el.get_text(strip=True) if title_el else ""
        price = _extract_number(str(price_el) if price_el else "")
        area = _extract_area(card.get_text())
        items.append({
            "platform": "LUN",
            "category": category,
            "title": title,
            "price_usd": price,
            "area_m2": area,
            "location": "Київ",
        })
    print(f"  знайдено карток: {len(items)}")
    return items


def _extract_number(text):
    if not text:
        return np.nan
    digits = re.sub(r"[^\d]", "", text.replace("\u00a0", ""))
    return float(digits) if digits else np.nan


def _extract_area(text):
    if not text:
        return np.nan
    m = re.search(r"(\d+[.,]?\d*)\s*(?:м²|кв\.?\s*м|m2)", text, re.IGNORECASE)
    return float(m.group(1).replace(",", ".")) if m else np.nan


SCRAPE_CONFIG = [
    (parse_olx, "https://www.olx.ua/uk/nedvizhimost/kvartiry/prodazha-kvartir/kiev/?currency=UAH&search%5Bfilter_enum_number_of_rooms_string%5D%5B0%5D=trehkomnatnye", "3-кімнатні квартири"),
    (parse_olx, "https://www.olx.ua/uk/nedvizhimost/doma/kiev/?currency=UAH", "будинки"),
    (parse_domria, "https://dom.ria.com/uk/search/?excludeSold=1&category=1&realty_type=2&operation=1&state_id=10&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&city_ids=10&limit=20&market=3&type=list&client=searchV2&ch=209_f_3,209_t_3,242_239,247_252", "3-кімнатні квартири"),
    (parse_domria, "https://dom.ria.com/uk/search/?excludeSold=1&category=4&realty_type=5&operation=1&state_id=10&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&city_ids=10&limit=20&market=3&type=list&client=searchV2&ch=226_223,242_239,247_252", "будинки"),
    (parse_lun, "https://lun.ua/uk/%D0%BA%D0%BE%D1%82%D0%B5%D0%B4%D0%B6%D1%96/%D0%BA%D0%B8%D1%97%D0%B2/%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81", "дуплекси"),
]


def scrape_all():
    all_items = []
    for func, url, category in SCRAPE_CONFIG:
        try:
            all_items.extend(func(url, category))
        except Exception as e:
            print(f"  [!] помилка парсингу {url}: {e}")
    df = pd.DataFrame(all_items)
    df.to_csv(os.path.join(OUT_DIR, "offers_raw.csv"), index=False,
              encoding="utf-8-sig")
    df.to_json(os.path.join(OUT_DIR, "offers_raw.json"),
               orient="records", force_ascii=False, indent=2)
    print(f"\nВсього пропозицій зібрано: {len(df)}")
    return df


def comparative_analysis(df):
    df = df.copy()
    df["price_per_m2"] = df["price_usd"] / df["area_m2"]

    print("\n" + "=" * 60)
    print("  I РІВЕНЬ. ПОРІВНЯЛЬНИЙ АНАЛІЗ КАТЕГОРІЙ")
    print("=" * 60)

    summary = df.groupby("category").agg(
        кількість=("price_usd", "count"),
        середня_ціна=("price_usd", "mean"),
        медіана_ціни=("price_usd", "median"),
        дисперсія_ціни=("price_usd", "var"),
        СКВ_ціни=("price_usd", "std"),
        середня_площа=("area_m2", "mean"),
        ціна_за_м2=("price_per_m2", "mean"),
    ).round(2)

    print(summary.to_string())
    summary.to_csv(os.path.join(OUT_DIR, "comparison_by_category.csv"),
                   encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    summary["середня_ціна"].plot(kind="bar", color="#4C72B0", alpha=0.85)
    plt.ylabel("Середня ціна, $")
    plt.title("Середня ціна за категоріями нерухомості (Київ)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_avg_price.png"), dpi=120)
    plt.show()

    plt.figure(figsize=(8, 5))
    cats = df["category"].unique()
    data = [df[df["category"] == c]["price_per_m2"].dropna() for c in cats]
    plt.boxplot(data, labels=cats)
    plt.ylabel("Ціна за м², $")
    plt.title("Розподіл ціни за квадратний метр")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_price_per_m2.png"), dpi=120)
    plt.show()

    plt.figure(figsize=(8, 5))
    for c in cats:
        plt.hist(df[df["category"] == c]["price_usd"].dropna(),
                 bins=15, alpha=0.5, label=c)
    plt.xlabel("Ціна, $")
    plt.ylabel("Кількість пропозицій")
    plt.title("Гістограма розподілу цін за категоріями")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_price_hist.png"), dpi=120)
    plt.show()

    return summary


UA_STOPWORDS = {
    "і", "й", "та", "в", "у", "на", "з", "із", "зі", "до", "по", "за", "що",
    "як", "це", "цей", "ця", "так", "але", "а", "о", "об", "від", "для", "при",
    "про", "над", "під", "не", "ні", "бо", "чи", "ще", "вже", "де", "там", "тут",
    "є", "був", "була", "було", "були", "буде", "ми", "ви", "вони", "він", "вона",
    "я", "ти", "мій", "ваш", "наш", "свій", "той", "теж", "або", "щоб", "тому",
}


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[а-яґєіїa-z0-9'’]+", text)
    return tokens


def remove_stopwords(tokens):
    return [t for t in tokens if t not in UA_STOPWORDS and len(t) > 2]


def lemmatize(tokens):
    doc = _NLP_SPACY(" ".join(tokens))
    return [tok.lemma_ for tok in doc]


_STEMMER = SnowballStemmer("russian")


def stem(tokens):
    return [_STEMMER.stem(t) for t in tokens]


def pos_analysis(text):
    doc = _NLP_SPACY(text)
    return [(tok.text, tok.pos_) for tok in doc]


def nlp_pipeline(text):
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    lemmas = lemmatize(tokens)
    stems = stem(tokens)
    return {"tokens": tokens, "lemmas": lemmas, "stems": stems}


POS_WORDS = {
    "якісний", "надійний", "чудовий", "вигідний", "зручний", "швидкий",
    "відмінний", "сучасний", "professional", "професійний", "рекомендую",
    "задоволений", "вчасно", "комфортний", "просторий", "новий", "хороший",
    "приємний", "оперативно", "порядний", "чесний",
}
NEG_WORDS = {
    "поганий", "жахливий", "обман", "затримка", "брак", "проблема", "дорогий",
    "розчарований", "повільний", "ненадійний", "тріщина", "недобудований",
    "шахрай", "невчасно", "недолік", "конфлікт", "скарга", "розчарування",
    "неякісний", "пошкодження",
}


def build_reviews():
    data = [
        ("Забудовник А", "Дуже задоволений, будинок зданий вчасно, якісний ремонт, рекомендую."),
        ("Забудовник А", "Сучасний просторий комфортний житловий комплекс, надійний забудовник."),
        ("Забудовник А", "Оперативно вирішили всі питання, порядний і чесний забудовник."),
        ("Забудовник А", "Зручне розташування, новий будинок, приємний персонал."),
        ("Забудовник Б", "Жахливий досвід, затримка здачі, тріщини на стінах, повний обман."),
        ("Забудовник Б", "Поганий менеджмент, недобудований будинок, дуже розчарований."),
        ("Забудовник Б", "Постійні проблеми з документами, конфлікт із підрядником."),
        ("Забудовник Б", "Ненадійний забудовник, невчасно, багато недоліків, скарга."),
        ("Забудовник В", "Загалом нормально, є дрібні недоліки, але ціна вигідна."),
        ("Забудовник В", "Зручне розташування, хоча ремонт неякісний місцями."),
        ("Забудовник В", "Сучасний проєкт, але дорогий, є проблема з паркінгом."),
        ("Забудовник В", "Хороший район, надійний, проте затримка з документами."),
    ]
    df = pd.DataFrame(data, columns=["developer", "text"])
    df.to_csv("reviews.csv", index=False, encoding="utf-8-sig")
    print(f"Сформовано відгуків: {len(df)} (збережено у reviews.csv)")
    return df


def analyze_reviews(reviews_df):
    print("\n" + "=" * 60)
    print("  II РІВЕНЬ. АНАЛІЗ ТОНАЛЬНОСТІ ВІДГУКІВ")
    print("=" * 60)

    results = []
    for _, row in reviews_df.iterrows():
        label, score = sentiment_lexicon(row["text"])
        results.append({**row.to_dict(), "sentiment": label, "score": score})
    res_df = pd.DataFrame(results)

    agg = res_df.groupby("developer")["score"].agg(["mean", "count"]).round(3)
    agg = agg.sort_values("mean", ascending=False)
    print("\nСередня тональність за забудовниками:")
    print(agg.to_string())

    res_df.to_csv(os.path.join(OUT_DIR, "sentiment_reviews.csv"),
                  index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 5))
    res_df["sentiment"].value_counts().plot(
        kind="bar", color=["#55A868", "#C44E52", "#8C8C8C"], alpha=0.85)
    plt.ylabel("Кількість відгуків")
    plt.title("Розподіл тональності відгуків про забудовників")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_sentiment.png"), dpi=120)
    plt.show()

    return res_df


def sentiment_lexicon(text):
    tokens = remove_stopwords(tokenize(text))
    lemmas = lemmatize(tokens)
    pos = sum(1 for w in lemmas if w in POS_WORDS)
    neg = sum(1 for w in lemmas if w in NEG_WORDS)
    total = pos + neg
    if total == 0:
        return "нейтральний", 0.0
    score = (pos - neg) / total
    if score > 0.2:
        label = "позитивний"
    elif score < -0.2:
        label = "негативний"
    else:
        label = "нейтральний"
    return label, round(score, 3)


def similarity_analysis(texts, labels):
    print("\n" + "=" * 60)
    print("  АНАЛІЗ ПОДІБНОСТІ ТЕКСТІВ (TF-IDF + cosine)")
    print("=" * 60)

    processed = [" ".join(nlp_pipeline(t)["lemmas"]) for t in texts]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(processed)
    sim_matrix = cosine_similarity(tfidf)

    sim_df = pd.DataFrame(sim_matrix.round(3), index=labels, columns=labels)
    print("\nМатриця косинусної подібності:")
    print(sim_df.to_string())
    sim_df.to_csv(os.path.join(OUT_DIR, "similarity_matrix.csv"),
                  encoding="utf-8-sig")

    plt.figure(figsize=(7, 6))
    plt.imshow(sim_matrix, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(label="Косинусна подібність")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Подібність текстів пропозицій")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_similarity.png"), dpi=120)
    plt.show()

    return sim_df


if __name__ == "__main__":
    print("############ ЛАБОРАТОРНА РОБОТА №5. Варіант 9 ############\n")

    offers = scrape_all()

    summary = comparative_analysis(offers)

    print("\n" + "=" * 60)
    print("  ДЕМОНСТРАЦІЯ NLP-КОНВЕЄРА")
    print("=" * 60)
    sample = "Сучасний просторий двокімнатний дуплекс у новому ЖК Києва."
    pipe = nlp_pipeline(sample)
    print("Текст:      ", sample)
    print("Токени:     ", pipe["tokens"])
    print("Леми:       ", pipe["lemmas"])
    print("Стеми:      ", pipe["stems"])
    print("POS-аналіз: ", pos_analysis(sample))

    reviews = build_reviews()
    sentiment_df = analyze_reviews(reviews)

    cat_texts, cat_labels = [], []
    for cat in offers["category"].unique():
        titles = offers[offers["category"] == cat]["title"].dropna().tolist()
        cat_texts.append(" ".join(titles[:30]))
        cat_labels.append(cat)
    if len(cat_texts) >= 2:
        similarity_analysis(cat_texts, cat_labels)

    print("\n############ РОБОТУ ЗАВЕРШЕНО. Результати у папці results/ ############")