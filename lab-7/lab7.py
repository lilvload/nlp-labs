import os
import json
import random
import string

import numpy as np

import nltk
from nltk.stem import SnowballStemmer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
    import whisper          # openai-whisper
    import torch
    STT_AVAILABLE = True
except Exception:
    STT_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

INTENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intents.json")
MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "air_alert_model.keras")
CONFIDENCE_THRESHOLD = 0.50
EPOCHS = 300
RECOGNITION_LANG = "uk"
WHISPER_MODEL = "small"
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
RECORD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "query.wav")

stemmer = SnowballStemmer("russian")


def preprocess_sentence(sentence):
    tokens = nltk.word_tokenize(sentence.lower(), language="russian")
    tokens = [stemmer.stem(t) for t in tokens if t not in string.punctuation]
    return tokens


def build_training_data(intents):
    words = []
    classes = []
    documents = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        if tag not in classes:
            classes.append(tag)
        for pattern in intent["patterns"]:
            tokens = preprocess_sentence(pattern)
            words.extend(tokens)
            documents.append((tokens, tag))

    words = sorted(set(words))
    classes = sorted(set(classes))

    X, y = [], []
    for tokens, tag in documents:
        bag = [1 if w in tokens else 0 for w in words]
        label = [0] * len(classes)
        label[classes.index(tag)] = 1
        X.append(bag)
        y.append(label)

    return np.array(X, dtype="float32"), np.array(y, dtype="float32"), words, classes

def build_model(input_size, output_size):
    model = keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(output_size, activation="softmax"),
    ])
    model.compile(
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        metrics=["accuracy"],
    )
    return model


def train_model(intents):
    X, y, words, classes = build_training_data(intents)
    model = build_model(len(X[0]), len(y[0]))
    print("------- навчання нейронної мережі -------")
    model.fit(X, y, epochs=EPOCHS, batch_size=8, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Точність на навчальній вибірці: {acc:.4f}, втрати: {loss:.4f}")
    model.save(MODEL_FILE)
    print(f"Модель збережено у файл: {MODEL_FILE}")
    return model, words, classes


def predict_intent(sentence, model, words, classes):
    tokens = preprocess_sentence(sentence)
    bag = np.array([[1 if w in tokens else 0 for w in words]], dtype="float32")
    probs = model.predict(bag, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def get_response(tag, intents):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Вибачте, сталася помилка."


def bot_answer(sentence, model, words, classes, intents):
    tag, confidence = predict_intent(sentence, model, words, classes)
    if confidence < CONFIDENCE_THRESHOLD:
        tag = "fallback"
    return get_response(tag, intents), tag, confidence


def init_tts():
    if not TTS_AVAILABLE:
        return None
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)
    for voice in engine.getProperty("voices"):
        name = (voice.name or "").lower()
        if "ukrain" in name or "russian" in name or "uk" in name:
            engine.setProperty("voice", voice.id)
            break
    return engine


def speak(engine, text):
    print(f"БОТ: {text}")
    if engine is not None:
        engine.say(text)
        engine.runAndWait()


def init_whisper():
    if not STT_AVAILABLE:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA доступна. Whisper працюватиме на GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA недоступна. Whisper працюватиме на CPU (повільніше).")
    print(f"Завантаження моделі Whisper '{WHISPER_MODEL}'...")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    return model


def record_audio():
    print(f"(Слухаю {RECORD_SECONDS} с... говоріть)")
    recording = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    wav_write(RECORD_FILE, SAMPLE_RATE, recording)
    return RECORD_FILE


def listen(whisper_model):
    audio_path = record_audio()
    use_fp16 = bool(STT_AVAILABLE and torch.cuda.is_available())
    result = whisper_model.transcribe(audio_path, language=RECOGNITION_LANG, fp16=use_fp16)
    text = (result.get("text") or "").strip()
    if text:
        print(f"ВИ (голос): {text}")
        return text
    print("(Не вдалося розпізнати мовлення)")
    return None


def text_loop(model, words, classes, intents, tts_engine=None):
    print("\n=== ТЕКСТОВИЙ РЕЖИМ. Введіть 'вихід' для завершення. ===")
    while True:
        user_input = input("ВИ: ").strip()
        if user_input.lower() in ("вихід", "exit", "quit"):
            speak(tts_engine, "Бережіть себе. До зв'язку.")
            break
        if not user_input:
            continue
        response, tag, conf = bot_answer(user_input, model, words, classes, intents)
        print(f"[намір: {tag}, впевненість: {conf:.2f}]")
        speak(tts_engine, response)


def voice_loop(model, words, classes, intents, tts_engine, whisper_model):
    if not STT_AVAILABLE or whisper_model is None:
        print("Whisper/мікрофон недоступні. Перемикаюся на текстовий режим.")
        text_loop(model, words, classes, intents, tts_engine)
        return

    print("\n=== ГОЛОСОВИЙ РЕЖИМ. Скажіть 'вихід', щоб завершити. ===")
    speak(tts_engine, "Голосовий режим активовано. Я бот підтримки під час повітряної тривоги. Поставте запитання.")
    while True:
        text = listen(whisper_model)
        if text is None:
            continue
        if "вихід" in text.lower():
            speak(tts_engine, "Бережіть себе. До зв'язку.")
            break
        response, tag, conf = bot_answer(text, model, words, classes, intents)
        print(f"[намір: {tag}, впевненість: {conf:.2f}]")
        speak(tts_engine, response)


def main():
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    with open(INTENTS_FILE, "r", encoding="utf-8") as f:
        intents = json.load(f)

    model, words, classes = train_model(intents)

    tts_engine = init_tts()

    print("\nОберіть режим роботи:")
    print("  1 - голосовий (мікрофон + Whisper STT + озвучення)")
    print("  2 - текстовий")
    mode = input("Ваш вибір [1/2]: ").strip()

    if mode == "1":
        whisper_model = init_whisper()
        voice_loop(model, words, classes, intents, tts_engine, whisper_model)
    else:
        text_loop(model, words, classes, intents, tts_engine)


if __name__ == "__main__":
    main()
