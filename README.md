# NLP Labs

Laboratory works on **Natural Language Processing**, written as standalone Python scripts.

The seven labs move from collecting and preprocessing real text to building models on top of it: web scraping of news, job postings, product reviews and real-estate listings; tokenization, lemmatization and stemming; part-of-speech analysis; classical text classification and clustering; lexicon-based and neural sentiment analysis; and, finally, a Ukrainian-language intent-based chatbot with optional speech recognition and speech synthesis.

> **Note:** most of the processed texts are in **Ukrainian** (news, Rozetka reviews, real-estate listings, chatbot intents). Lab 2 works with English job postings and Lab 3 compares Ukrainian and English texts. Code comments, console output and chart labels are in Ukrainian.

## Contents

| Lab | Topic | Main tools |
|-----|-------|------------|
| [Lab 1](lab-1/lab_work_1.py) | News monitoring: term frequencies, trend and forecast | requests, BeautifulSoup, NumPy, WordCloud |
| [Lab 2](lab-2/main.py) | Text preprocessing pipeline on NLP Engineer job postings | NLTK, spaCy, langdetect |
| [Lab 3](lab-3/lab3.py) | Part-of-speech frequencies vs. text topic (Ukrainian and English) | NLTK, pymorphy3 |
| [Lab 4](lab-4/lab4.py) | Review classification, frequency analysis and clustering | scikit-learn, SciPy, NLTK |
| [Lab 5](lab-5/lab_work_5.py) | Real-estate listings: comparative analysis, sentiment and text similarity | spaCy, pymorphy3, scikit-learn |
| [Lab 6](lab-6/lab_work_6.py) | Neural networks for listing categories and review sentiment | TensorFlow / Keras, scikit-learn |
| [Lab 7](lab-7/lab7.py) | Voice chatbot for air-raid alert guidance | TensorFlow / Keras, NLTK, Whisper, pyttsx3 |

## Repository structure

```
nlp-labs/
├── lab-1/   lab_work_1.py, results/ (monitoring.csv, top3_daily.csv, charts)
├── lab-2/   main.py, output/ (intermediate texts of every pipeline stage, CSV, charts, summary JSON)
├── lab-3/   lab3.py, texts/ (uk_texts.py, en_texts.py), results/ (CSV, charts, console_output.txt)
├── lab-4/   lab4.py, results/ (charts)
├── lab-5/   lab_work_5.py, reviews.csv, results/ (scraped offers, CSV, charts)
├── lab-6/   lab_work_6.py, data/ (scraped offers and reviews, charts)
└── lab-7/   lab7.py, intents.json, air_alert_model.keras, query.wav
```

Each folder is self-contained, and the results of a previous run (data files and charts) are committed next to the script, so they can be viewed on GitHub without running anything.

## Labs

### Lab 1. News monitoring: term frequencies, trend and forecast

- Scrapes the RSS feeds of two Ukrainian news sites (Ukrainska Pravda and Ukrinform) and prints the top 5 terms of the current snapshot.
- Cleans the text with regular expressions (HTML tags, links, entities, non-Cyrillic characters) and removes a custom list of Ukrainian stop words.
- Analyses a week of monitoring (`results/monitoring.csv`: 7 days × 3 snapshots per day, morning / noon / evening, top 5 terms in each).
- Computes the mean, variance and standard deviation of the total term frequency.
- Fits a linear trend with the **least squares method** (implemented in NumPy), reports R² and forecasts one week ahead.
- Builds a **word cloud** for the whole period, finds the three most frequent terms and forecasts the daily dynamics of each of them.

### Lab 2. Text preprocessing pipeline on NLP Engineer job postings

Answers the question "what are the dominant requirements for an NLP Engineer?".

- Collects job descriptions from public job-board APIs (RemoteOK, with Arbeitnow and Himalayas as fallbacks) and keeps only English texts (`langdetect`).
- Runs a full preprocessing pipeline and saves the text after every stage to `output/`: filtering, normalization, tokenization in **three ways** (NLTK `word_tokenize`, `wordpunct_tokenize` and the spaCy tokenizer), stop-word removal, POS-aware **lemmatization** (WordNet) and **stemming** (Porter).
- Builds the top 10 keywords by lemmas and by stems.
- Counts mentions of technical skills from a hand-made dictionary (Python, PyTorch, transformers, LLMs, Docker, cloud, MLOps and others) and plots the dominant requirements.

On the saved run, 32 job postings were analysed (about 20.8k tokens, 12k after stop-word removal); cloud platforms, communication and MLOps / CI-CD were the most mentioned requirements.

### Lab 3. Part-of-speech frequencies vs. text topic

Studies how the distribution of parts of speech depends on the topic of a text.

- Uses texts on four topics (mathematics, physics, art and fiction) in **Ukrainian** and **English**, at least 5,000 characters each (`texts/`).
- Tags English texts with NLTK (`universal` tagset) and Ukrainian texts with **pymorphy3**, mapping OpenCorpora tags to the same universal tags.
- Computes the percentage of each tag per text and saves summary tables (`freq_uk.csv`, `freq_en.csv`).
- Visualizes the results as grouped bar charts and heatmaps for both languages.

### Lab 4. Review classification, frequency analysis and clustering

Works with real customer reviews of five products scraped from the **Rozetka** API (iPhone, Lenovo laptop, JBL headphones, Philips coffee machine, Xiaomi robot vacuum).

- Turns star ratings into labels (4–5 positive, 1–2 negative, 3 dropped), cleans and tokenizes the Ukrainian texts and removes stop words.
- **Supervised classification:** TF-IDF (unigrams and bigrams) with Logistic Regression and Multinomial Naive Bayes; reports accuracy, precision, recall and F1, checks the "accuracy > 60%" requirement and plots the confusion matrix of the best model.
- **Frequency analysis:** top 15 terms by TF-IDF, lexical dispersion plot, word-length distribution (mean and variance) and the top 12 bigrams.
- **Unsupervised clustering:** K-Means with the number of clusters chosen by the silhouette score, characteristic words of each cluster, a Ward dendrogram and a 2D PCA projection of the clusters.

### Lab 5. Real-estate listings: comparative analysis, sentiment and text similarity

Analyses the Kyiv real-estate market in three categories (3-room flats, houses and duplexes).

- Scrapes listings from **OLX**, **DOM.RIA** and **LUN**, extracting titles, prices and floor area with regular expressions.
- **Level I, comparative analysis:** count, mean, median, variance, standard deviation, average area and price per m² for each category, with bar, box and histogram charts.
- Demonstrates a Ukrainian NLP pipeline: tokenization, stop-word removal, lemmatization (spaCy `uk_core_news_sm`), stemming (Snowball) and POS tagging.
- **Level II, sentiment analysis:** a lexicon-based classifier of reviews about three developers (positive and negative word lists matched against lemmas), with the average score per developer.
- **Text similarity:** TF-IDF on lemmatized listing titles and a cosine similarity matrix between the categories.

### Lab 6. Neural networks for listing categories and review sentiment

Revisits the task of Lab 5 with neural networks.

- Scrapes listings for the same three categories from LUN, DOM.RIA and OLX, and reviews of developers from the same platforms (`data/`).
- **Level I:** classifies listings into categories with a dense neural network (Keras) on TF-IDF features; plots training accuracy and the confusion matrix, and compares price statistics across categories.
- **Level II:** classifies review sentiment with an Embedding + GlobalAveragePooling network and computes the share of positive reviews per developer.

### Lab 7. Voice chatbot for air-raid alert guidance

A Ukrainian-language support bot that tells the user what to do during an air-raid alert.

- `intents.json` defines 12 intents: greeting, alert started, shelter, the "two walls" rule, what to take, alert duration, all clear, being in transport, emergency numbers, calming down, goodbye and a fallback.
- Patterns are tokenized and stemmed (NLTK, Snowball), converted to a **bag of words**, and a dense neural network (Keras) is trained to classify the intent; the model is saved to `air_alert_model.keras`.
- If the model's confidence is below 0.5, the bot gives the fallback answer.
- Works in two modes: **text** (console chat) and **voice**, where speech is recorded from the microphone, recognized with **OpenAI Whisper** (on GPU if CUDA is available) and the answer is spoken with **pyttsx3**. Without the speech libraries the bot falls back to text mode.

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/lilvload/nlp-labs.git
   cd nlp-labs
   ```

2. Create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate

   pip install numpy pandas matplotlib scipy scikit-learn requests beautifulsoup4 lxml wordcloud \
               nltk spacy langdetect pymorphy3 pymorphy3-dicts-uk tensorflow
   python -m spacy download uk_core_news_sm
   ```

   `lxml` is needed to parse the RSS feeds in Lab 1. For the voice mode of Lab 7 also install:

   ```bash
   pip install openai-whisper torch sounddevice pyttsx3
   ```

3. Run a script from its own folder, because the scripts read data and save results by relative path:

   ```bash
   cd lab-3
   python lab3.py
   ```

### Notes on running

- **Labs 1, 2, 4, 5 and 6** download live data from websites and public APIs, so they need an internet connection, and a new run will give different results from the committed ones. The scrapers depend on the current page layout of each site and may stop finding data if the layout changes.
- **Lab 1** scrapes only the current news snapshot; the weekly trend and forecast are computed from the committed `results/monitoring.csv`.
- **Lab 4** saves its charts to the current folder, not to `results/`.
- **Lab 7** is interactive: it retrains the model on every start and then asks you to choose the text or voice mode. Type `вихід` (or `exit`) to quit.

## Technologies

Python · NLTK · spaCy · pymorphy3 · scikit-learn · TensorFlow / Keras · OpenAI Whisper · pyttsx3 · pandas · NumPy · Matplotlib · BeautifulSoup · WordCloud
