import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import nltk
from nltk import word_tokenize, pos_tag
import pymorphy3

from texts.uk_texts import UK_TEXTS
from texts.en_texts import EN_TEXTS

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

UNIVERSAL_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
                  "ADP", "CONJ", "NUM", "PRT", "X", "PUNCT"]

UK_TAG_LABELS = {
    "NOUN": "іменник", "VERB": "дієслово", "ADJ": "прикметник",
    "ADV": "прислівник", "PRON": "займенник", "DET": "детермінатив",
    "ADP": "прийменник", "CONJ": "сполучник", "NUM": "числівник",
    "PRT": "частка", "X": "інше", "PUNCT": "пунктуація",
}

PYMORPHY_TO_UNIVERSAL = {
    "NOUN": "NOUN",   # іменник
    "ADJF": "ADJ",    # повний прикметник
    "ADJS": "ADJ",    # короткий прикметник
    "COMP": "ADJ",    # компаратив
    "VERB": "VERB",   # дієслово (особова форма)
    "INFN": "VERB",   # інфінітив
    "PRTF": "VERB",   # дієприкметник повний
    "PRTS": "VERB",   # дієприкметник короткий
    "GRND": "VERB",   # дієприслівник
    "NUMR": "NUM",    # числівник
    "ADVB": "ADV",    # прислівник
    "NPRO": "PRON",   # займенник-іменник
    "PRED": "ADV",    # предикатив
    "PREP": "ADP",    # прийменник
    "CONJ": "CONJ",   # сполучник
    "PRCL": "PRT",    # частка
    "INTJ": "X",      # вигук
}


def tag_english(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens, tagset="universal")  # universal tagset
    result = []
    for tok, tag in tagged:
        if re.fullmatch(r"\W+", tok):      # розділові знаки -> PUNCT
            tag = "PUNCT"
        result.append((tok, tag))
    return result


_morph = pymorphy3.MorphAnalyzer(lang="uk")

def tag_ukrainian(text):
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    result = []
    for tok in tokens:
        if re.fullmatch(r"\W+", tok):      # пунктуація
            result.append((tok, "PUNCT"))
            continue
        parsed = _morph.parse(tok)[0]
        pos = parsed.tag.POS               # тег OpenCorpora
        uni = PYMORPHY_TO_UNIVERSAL.get(pos, "X")
        result.append((tok, uni))
    return result


def tag_frequencies(tagged, exclude_punct=True):
    counts = {t: 0 for t in UNIVERSAL_TAGS}
    for _, tag in tagged:
        counts[tag] = counts.get(tag, 0) + 1
    if exclude_punct:
        counts.pop("PUNCT", None)
    total = sum(counts.values())
    if total == 0:
        return {t: 0.0 for t in counts}
    return {t: round(100.0 * c / total, 2) for t, c in counts.items()}


def print_stats(title, tagged):
    freq = tag_frequencies(tagged)
    n_tokens = len(tagged)
    n_signif = sum(1 for _, t in tagged if t != "PUNCT")
    print("------------", title, "-------------")
    print("усього токенів:", n_tokens, "| значущих (без пунктуації):", n_signif)
    for t in UNIVERSAL_TAGS:
        if t == "PUNCT":
            continue
        if freq.get(t, 0) > 0:
            print(f"   {t:5s} ({UK_TAG_LABELS[t]:12s}) = {freq[t]:5.2f} %")
    print("-----------------------------------------------------")
    return freq


def plot_grouped_bars(df, lang, fname):
    main_tags = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "CONJ", "PRT"]
    sub = df[main_tags]
    ax = sub.T.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_ylabel("Частота тегу, %")
    ax.set_xlabel("POS-тег")
    ax.set_title(f"Залежність частоти POS-тегів від тематики ({lang})")
    ax.legend(title="Тематика")
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


def plot_heatmap(df, lang, fname):
    main_tags = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET",
                 "ADP", "CONJ", "NUM", "PRT"]
    data = df[main_tags].values
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(main_tags)))
    ax.set_xticklabels(main_tags)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)
    ax.set_title(f"Теплова карта частот POS-тегів ({lang})")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="Частота, %")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, fname)
    plt.savefig(path, dpi=130)
    plt.close()
    return path


if __name__ == "__main__":

    theme_order_uk = ["математика", "фізика", "мистецтво", "художні_твори"]
    theme_order_en = ["mathematics", "physics", "art", "fiction"]
    theme_titles   = ["Математика", "Фізика", "Мистецтво", "Художні твори"]

    print("\n################  АНАЛІЗ УКРАЇНОМОВНИХ ТЕКСТІВ  ################\n")
    uk_rows = {}
    for theme, title in zip(theme_order_uk, theme_titles):
        tagged = tag_ukrainian(UK_TEXTS[theme])
        freq = print_stats(f"UK / {title}", tagged)
        uk_rows[title] = freq
    df_uk = pd.DataFrame(uk_rows).T[[t for t in UNIVERSAL_TAGS if t != "PUNCT"]]

    print("\n################  АНАЛІЗ АНГЛОМОВНИХ ТЕКСТІВ  ################\n")
    en_rows = {}
    for theme, title in zip(theme_order_en, theme_titles):
        tagged = tag_english(EN_TEXTS[theme])
        freq = print_stats(f"EN / {title}", tagged)
        en_rows[title] = freq
    df_en = pd.DataFrame(en_rows).T[[t for t in UNIVERSAL_TAGS if t != "PUNCT"]]

    print("\n=== Зведена таблиця частот тегів (українська), % ===")
    print(df_uk.round(2).to_string())
    print("\n=== Зведена таблиця частот тегів (англійська), % ===")
    print(df_en.round(2).to_string())

    df_uk.round(2).to_csv(os.path.join(OUT_DIR, "freq_uk.csv"),
                          encoding="utf-8-sig")
    df_en.round(2).to_csv(os.path.join(OUT_DIR, "freq_en.csv"),
                          encoding="utf-8-sig")

    p1 = plot_grouped_bars(df_uk, "українська", "bars_uk.png")
    p2 = plot_grouped_bars(df_en, "англійська", "bars_en.png")
    p3 = plot_heatmap(df_uk, "українська", "heatmap_uk.png")
    p4 = plot_heatmap(df_en, "англійська", "heatmap_en.png")

    print("\nЗбережені файли результатів:")
    for p in [p1, p2, p3, p4,
              os.path.join(OUT_DIR, "freq_uk.csv"),
              os.path.join(OUT_DIR, "freq_en.csv")]:
        print("  ", p)
