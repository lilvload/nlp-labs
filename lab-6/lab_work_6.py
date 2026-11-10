import os
import re
import time
import random

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/120.0 Safari/537.36'),
    'Accept-Language': 'uk-UA,uk;q=0.9,en;q=0.8',
}

OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


PLATFORM_URLS = {
    '3-кімнатні квартири': {
        'LUN':     'https://lun.ua/sale/kyiv/flats?srsltid=AfmBOorO3qcgTipLSmES7YubHjXivo4Xe0538hs2wVytCADskGzazNyF&room_count=3',
        'DOM_RIA': 'https://dom.ria.com/uk/search?excludeSold=1&category=1&realty_type=2&operation=1&state_id=10&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&city_ids=10&client=searchV2&limit=20&type=list&ch=209_f_3,209_t_3,242_239,247_252',
        'OLX':     'https://www.olx.ua/uk/nedvizhimost/kvartiry/prodazha-kvartir/kiev/?currency=UAH&search%5Bfilter_enum_number_of_rooms_string%5D%5B0%5D=trehkomnatnye',
    },
    'будинки': {
        'LUN':     'https://lun.ua/sale/kyiv/houses',
        'DOM_RIA': 'https://dom.ria.com/uk/search?excludeSold=1&category=4&realty_type=5&operation=1&state_id=10&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&city_ids=10&client=searchV2&limit=20&type=list&ch=226_223,242_239,247_252',
        'OLX':     'https://www.olx.ua/uk/nedvizhimost/doma/prodazha-domov/kiev/?currency=UAH',
    },
    'дуплекси': {
        'LUN':     'https://lun.ua/uk/%D0%BA%D0%BE%D1%82%D0%B5%D0%B4%D0%B6%D1%96/%D0%BA%D0%B8%D1%97%D0%B2/%D0%B4%D1%83%D0%BF%D0%BB%D0%B5%D0%BA%D1%81?srsltid=AfmBOoqyBzQHXcVET3udXFw0eDNXu5fy8laYrcDNjlNcrYnAD2QzSIYl',
        'DOM_RIA': 'https://dom.ria.com/uk/search?excludeSold=1&category=4&realty_type=8&operation=1&state_id=10&price_cur=1&wo_dupl=1&sort=inspected_sort&firstIteraction=false&city_ids=10&client=searchV2&limit=20&type=list&ch=226_223,242_239,247_252',
        'OLX':     'https://www.olx.ua/uk/nedvizhimost/doma/prodazha-domov/dupleks/kiev/',
    },
}

REVIEW_URLS = {
    'LUN':     'https://lun.ua/uk/developers/kiev',
    'DOM_RIA': 'https://dom.ria.com/uk/realtors/?type=developer&city_id=10',
    'OLX':     'https://www.olx.ua/uk/nedvizhimost/kvartiry/prodazha-kvartir/kiev/',
}

OFFER_SELECTORS = {
    'LUN': {
        'card':  'div[class*="card"]',
        'title': 'div[class*="title"], span[class*="name"]',
        'price': 'div[class*="price"]',
        'desc':  'div[class*="desc"], div[class*="info"]',
    },
    'DOM_RIA': {
        'card':  'section.ticket-clear, div[class*="ticket"]',
        'title': 'a.realty-link, span[class*="title"], h3',
        'price': 'b.size22, span[class*="price"], div[class*="price"]',
        'desc':  'p, div[class*="description"]',
    },
    'OLX': {
        'card':  'div[data-cy="l-card"]',
        'title': 'h6, h4',
        'price': 'p[data-testid="ad-price"]',
        'desc':  'p[data-testid="ad-description"]',
    },
}

REVIEW_SELECTORS = {
    'LUN': {
        'card':   'div[class*="developer"], div[class*="card"]',
        'dev':    'div[class*="name"], h3, h2',
        'text':   'div[class*="review"], div[class*="comment"], p',
        'rating': 'div[class*="rating"], span[class*="stars"]',
    },
    'DOM_RIA': {
        'card':   'div[class*="realtor"], div[class*="card"]',
        'dev':    'a[class*="name"], h3, h2',
        'text':   'div[class*="review"], div[class*="comment"], p',
        'rating': 'span[class*="rating"], div[class*="rating"]',
    },
    'OLX': {
        'card':   'div[data-cy="l-card"]',
        'dev':    'h6, h4',
        'text':   'p[data-testid="ad-description"], p',
        'rating': 'span[class*="rating"]',
    },
}


def fetch_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        time.sleep(random.uniform(1.0, 2.5))
        return resp.text
    except requests.RequestException as e:
        print(f'[!] Помилка завантаження {url}: {e}')
        return None


def parse_offers(html, platform_name, category):
    offers = []
    if not html:
        return offers

    sel = OFFER_SELECTORS[platform_name]
    soup = BeautifulSoup(html, 'html.parser')
    cards = soup.select(sel['card'])
    for card in cards:
        title_el = card.select_one(sel['title'])
        price_el = card.select_one(sel['price'])
        desc_el  = card.select_one(sel['desc'])

        title       = title_el.get_text(strip=True) if title_el else ''
        price_raw   = price_el.get_text(strip=True) if price_el else ''
        description = desc_el.get_text(strip=True)  if desc_el  else ''

        price_digits = re.sub(r'[^\d]', '', price_raw)
        price = int(price_digits) if price_digits else None

        if title:
            offers.append({
                'platform':    platform_name,
                'category':    category,
                'title':       title,
                'price':       price,
                'description': description,
            })
    return offers


def scrape_all():
    all_offers = []
    for category, platforms in PLATFORM_URLS.items():
        for platform_name, url in platforms.items():
            print(f'[*] Скрапінг {platform_name} -> {category}')
            html = fetch_page(url)
            offers = parse_offers(html, platform_name, category)
            print(f'    знайдено пропозицій: {len(offers)}')
            all_offers.extend(offers)

    df = pd.DataFrame(all_offers)
    df.to_csv(os.path.join(OUTPUT_DIR, 'offers.csv'), index=False, encoding='utf-8-sig')
    df.to_json(os.path.join(OUTPUT_DIR, 'offers.json'), orient='records', force_ascii=False, indent=2)
    print(f'[+] Збережено: {OUTPUT_DIR}/offers.csv, {OUTPUT_DIR}/offers.json')
    return df


def parse_reviews(html, platform_name):
    parsed = []
    if not html:
        return parsed

    sel = REVIEW_SELECTORS[platform_name]
    soup = BeautifulSoup(html, 'html.parser')
    cards = soup.select(sel['card'])
    for card in cards:
        dev_el  = card.select_one(sel['dev'])
        text_el = card.select_one(sel['text'])
        rate_el = card.select_one(sel['rating'])

        developer  = dev_el.get_text(strip=True)  if dev_el  else ''
        text       = text_el.get_text(strip=True) if text_el else ''
        rating_raw = rate_el.get_text(strip=True) if rate_el else ''

        rating_digits = re.sub(r'[^\d.]', '', rating_raw)
        try:
            rating = float(rating_digits)
        except ValueError:
            rating = None

        label = None
        if rating is not None:
            label = 1 if rating >= 3.5 else 0

        if developer and text:
            parsed.append({
                'developer': developer,
                'text':      text,
                'rating':    rating,
                'label':     label,
                'platform':  platform_name,
            })
    return parsed


def scrape_reviews():
    all_reviews = []
    for platform_name, url in REVIEW_URLS.items():
        print(f'[*] Скрапінг відгуків {platform_name}')
        html = fetch_page(url)
        reviews = parse_reviews(html, platform_name)
        print(f'    знайдено відгуків: {len(reviews)}')
        all_reviews.extend(reviews)

    df = pd.DataFrame(all_reviews)
    df.to_csv(os.path.join(OUTPUT_DIR, 'reviews.csv'), index=False, encoding='utf-8-sig')
    print(f'[+] Збережено: {OUTPUT_DIR}/reviews.csv  ({len(df)} відгуків)')
    return df


def comparative_analysis(df):
    df = df.dropna(subset=['title']).copy()
    df['text'] = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.lower()

    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text']).toarray()

    cat_to_idx = {c: i for i, c in enumerate(sorted(df['category'].unique()))}
    y = df['category'].map(cat_to_idx).values
    num_classes = len(cat_to_idx)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = Sequential([
        Dense(256, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        validation_split=0.15,
                        epochs=20, batch_size=16, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n[I рівень] Точність класифікації категорій: {acc:.4f}')

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred,
                                target_names=list(cat_to_idx.keys()),
                                zero_division=0))

    plt.figure()
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.title('Навчання ШНМ: класифікація категорій нерухомості')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_training.png'), dpi=120, bbox_inches='tight')
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(range(num_classes), list(cat_to_idx.keys()), rotation=45, ha='right')
    plt.yticks(range(num_classes), list(cat_to_idx.keys()))
    plt.xlabel('Прогноз')
    plt.ylabel('Факт')
    plt.title('Матриця сплутування категорій')
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_confusion.png'), dpi=120, bbox_inches='tight')
    plt.show()

    price_stats = df.groupby('category')['price'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
    print('\nСтатистика цін за категоріями:')
    print(price_stats)

    plt.figure()
    df.boxplot(column='price', by='category', rot=15)
    plt.suptitle('')
    plt.title('Розподіл цін за категоріями нерухомості')
    plt.ylabel('Ціна')
    plt.savefig(os.path.join(OUTPUT_DIR, 'category_prices.png'), dpi=120, bbox_inches='tight')
    plt.show()

    return model, acc, price_stats


def sentiment_analysis(reviews_df):
    reviews_df = reviews_df.dropna(subset=['text', 'label']).copy()
    texts  = reviews_df['text'].astype(str).tolist()
    labels = reviews_df['label'].astype(int).values

    max_words = 5000
    max_len   = 60
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    model = Sequential([
        Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train,
                        validation_split=0.15,
                        epochs=15, batch_size=16, verbose=1)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n[II рівень] Точність аналізу тональності: {acc:.4f}')

    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
    print(classification_report(y_test, y_pred,
                                target_names=['негатив', 'позитив'],
                                zero_division=0))

    plt.figure()
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.xlabel('Епоха')
    plt.ylabel('Точність')
    plt.title('Навчання ШНМ: аналіз тональності відгуків')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_training.png'), dpi=120, bbox_inches='tight')
    plt.show()

    reviews_df['pred'] = (model.predict(X) > 0.5).astype(int).flatten()
    by_dev = reviews_df.groupby('developer')['pred'].agg(['count', 'mean'])
    by_dev.rename(columns={'mean': 'positive_ratio'}, inplace=True)
    print('\nЧастка позитивних відгуків за забудовниками:')
    print(by_dev)

    plt.figure()
    by_dev['positive_ratio'].sort_values().plot(kind='barh')
    plt.xlabel('Частка позитивних відгуків')
    plt.title('Тональність відгуків про забудовників')
    plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_by_developer.png'), dpi=120, bbox_inches='tight')
    plt.show()

    return model, acc, by_dev


if __name__ == '__main__':
    print('=' * 70)
    print('ЕТАП 1. WEB-СКРАПІНГ ПЛАТФОРМ НЕРУХОМОСТІ')
    print('=' * 70)
    offers_df = scrape_all()

    print('\n' + '=' * 70)
    print('ЕТАП 2. ПОРІВНЯЛЬНИЙ АНАЛІЗ КАТЕГОРІЙ (I РІВЕНЬ)')
    print('=' * 70)
    if not offers_df.empty and len(offers_df['category'].unique()) > 1:
        comparative_analysis(offers_df)
    else:
        print('[!] Недостатньо даних для класифікації категорій.')

    print('\n' + '=' * 70)
    print('ЕТАП 3. АНАЛІЗ ТОНАЛЬНОСТІ ВІДГУКІВ ПРО ЗАБУДОВНИКІВ (II РІВЕНЬ)')
    print('=' * 70)
    reviews_df = scrape_reviews()
    if not reviews_df.empty and reviews_df['label'].notna().any():
        sentiment_analysis(reviews_df)
    else:
        print('[!] Аналіз тональності пропущено: скрапінг не повернув розмічених відгуків.')

    print('\n[✓] Виконання програми завершено.')