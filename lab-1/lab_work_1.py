import os
import re
import json
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from collections import Counter
from wordcloud import WordCloud

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


NEWS_SITES = {
    "Українська правда": "https://www.pravda.com.ua/rss/",
    "Укрінформ": "https://www.ukrinform.ua/rss",
}

STOP_WORDS = set("""
і й та але або що як це той ця те ці так вже ще де коли куди звідки
у в на з зі із до від про за над під біля між для без при через о об
не ні ані то чи бо аби якщо щоб хоч хай нехай який яка яке які чий чия
його її їх йому їй нам вам мені тобі собі себе він вона воно вони ми ви ти я
а же б би ж лише тільки навіть саме адже отже тому потім тоді зараз нині
є був була було були буде будуть має мають можна треба слід щодо також
один два три рік року році роках день дня році час раз разів усе все всі весь
""".split())


def scrape_news(name, url):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    texts = []
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "xml")
        for item in soup.find_all("item"):
            title = item.title.get_text(strip=True) if item.title else ""
            descr = item.description.get_text(strip=True) if item.description else ""
            texts.append(f"{title}. {descr}")
        print(f"[OK]  {name}: отримано {len(texts)} новин")
    except Exception as e:
        print(f"[ERR] {name}: {e}")
    return texts


def clean_text(raw_text):
    text = re.sub(r"<[^>]+>", " ", raw_text)            # HTML-теги
    text = re.sub(r"https?://\S+", " ", text)           # посилання
    text = re.sub(r"&[a-z]+;", " ", text)               # HTML-entities
    text = text.lower()
    text = re.sub(r"[^а-яіїєґ'\s]", " ", text)          # лишаємо тільки укр. літери
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_filter(text):
    words = text.split()
    return [w for w in words if w not in STOP_WORDS and len(w) >= 4]


def top_terms(texts, top_n=5):
    counter = Counter()
    for t in texts:
        counter.update(tokenize_filter(clean_text(t)))
    return counter.most_common(top_n)


def mnk_linear(y):
    n = len(y)
    t = np.arange(n)
    F = np.column_stack([np.ones(n), t])
    coef = np.linalg.inv(F.T @ F) @ F.T @ y
    y_hat = F @ coef
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 1.0
    return coef, y_hat, r2


def mnk_forecast(y, steps):
    coef, y_hat, r2 = mnk_linear(y)
    n = len(y)
    t_future = np.arange(n, n + steps)
    F_future = np.column_stack([np.ones(steps), t_future])
    y_future = F_future @ coef
    return y_hat, y_future, coef, r2


def stat_characteristics(y, label):
    m = np.mean(y)
    d = np.var(y)
    s = np.sqrt(d)
    print(f"--- Статистичні характеристики: {label} ---")
    print(f"    мат. сподівання = {m:.4f}")
    print(f"    дисперсія       = {d:.4f}")
    print(f"    СКВ             = {s:.4f}")
    return m, d, s


if __name__ == "__main__":
    print("=== Скрапінг новинних сайтів ===")
    all_now = []
    for site, url in NEWS_SITES.items():
        all_now += scrape_news(site, url)

    print("\nТоп-5 термінів поточного зрізу:")
    for term, freq in top_terms(all_now, 5):
        print(f"   {term}: {freq}")


    df = pd.read_csv(os.path.join(OUT_DIR, "monitoring.csv"))
    print("\n=== Аналіз тижневого моніторингу (monitoring.csv) ===")
    print(df.head(10).to_string(index=False))

    sums = df.groupby(["День", "Час"])["Частота"].sum().reset_index()
    order = {"Ранок": 0, "Обід": 1, "Вечір": 2}
    sums["ord"] = sums["Час"].map(order)
    sums = sums.sort_values(["День", "ord"]).reset_index(drop=True)
    y_sum = sums["Сума_частот"].values if "Сума_частот" in sums else sums["Частота"].values

    stat_characteristics(y_sum, "Сума частот топ-5 (всі зрізи)")
    y_hat, y_fc, coef, r2 = mnk_forecast(y_sum, steps=21)  # прогноз на тиждень (7дн*3зрізи)
    print(f"    Тренд: y = {coef[0]:.2f} + {coef[1]:.3f}*t,  R^2 = {r2:.4f}")

    plt.figure(figsize=(11, 5))
    plt.plot(range(len(y_sum)), y_sum, "o-", label="Сума частот (факт)", color="#1f77b4")
    plt.plot(range(len(y_hat)), y_hat, "--", label=f"Тренд МНК (R²={r2:.3f})", color="#ff7f0e")
    fx = range(len(y_sum), len(y_sum) + len(y_fc))
    plt.plot(fx, y_fc, "s--", label="Прогноз на тиждень", color="#2ca02c")
    plt.axvline(len(y_sum) - 0.5, color="grey", ls=":", alpha=0.7)
    plt.xlabel("Часовий зріз (ранок/обід/вечір)")
    plt.ylabel("Сумарна частота топ-5 термінів")
    plt.title("Динаміка сумарної частоти термінів: тренд і прогноз (МНК)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_trend_forecast.png"), dpi=130)
    plt.close()

    freq_dict = df.groupby("Термін")["Частота"].sum().to_dict()
    wc = WordCloud(width=900, height=450, background_color="white",
                   colormap="viridis").generate_from_frequencies(freq_dict)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Хмара термінів за весь період моніторингу")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_wordcloud.png"), dpi=130)
    plt.close()

    top3 = df.groupby("Термін")["Частота"].sum().nlargest(3).index.tolist()
    print(f"\n*** Три найчастіші терміни за весь період: {top3}")

    daily = (df[df["Термін"].isin(top3)]
             .groupby(["День", "Термін"])["Частота"].sum()
             .unstack(fill_value=0)
             .reindex(columns=top3))
    daily.to_csv(os.path.join(OUT_DIR, "top3_daily.csv"), encoding="utf-8-sig")

    plt.figure(figsize=(11, 5.5))
    colors = ["#d62728", "#9467bd", "#8c564b"]
    days = np.arange(len(daily))
    future_days = np.arange(len(daily), len(daily) + 7)
    for k, term in enumerate(top3):
        y = daily[term].values.astype(float)
        yh, yf, c, r = mnk_forecast(y, 7)
        plt.plot(days, y, "o-", color=colors[k], label=f"{term} (факт)")
        plt.plot(days, yh, "--", color=colors[k], alpha=0.6)
        plt.plot(future_days, yf, "s--", color=colors[k], alpha=0.9,
                 label=f"{term}: прогноз (R²={r:.2f})")
        print(f"    Тренд '{term}': y={c[0]:.2f}+{c[1]:.3f}*t, R²={r:.3f}")
    plt.axvline(len(daily) - 0.5, color="grey", ls=":", alpha=0.7)
    plt.xlabel("День моніторингу")
    plt.ylabel("Добова сумарна частота терміну")
    plt.title("Тренд і прогноз динаміки 3 найчастіших термінів (МНК)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_top3_forecast.png"), dpi=130)
    plt.close()

    print("\n=== Готово. Результати збережено у папці 'results/' ===")
