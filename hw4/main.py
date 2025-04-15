import os
import re
import html
import math
from collections import Counter, defaultdict
from bs4 import BeautifulSoup

PAGES_DIR = '../hw4/pages/'
TOKENS_DIR = '../hw4/input/tokens/'
LEMMAS_DIR = '../hw4/input/lemmas/'
OUTPUT_TOKENS = 'output/tokens_tf_idf/'
OUTPUT_LEMMAS = 'output/lemmas_tf_idf/'

os.makedirs(PAGES_DIR, exist_ok=True)
os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(LEMMAS_DIR, exist_ok=True)
os.makedirs(OUTPUT_TOKENS, exist_ok=True)
os.makedirs(OUTPUT_LEMMAS, exist_ok=True)


def preprocess_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.decompose()
    return soup.get_text(separator=' ', strip=True)


def normalize_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^а-яё\s-]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def compute_tf(word_counts, total_words):
    return {word: count / total_words for word, count in word_counts.items()}


def compute_idf(doc_freq, total_docs):
    return {word: math.log(total_docs / (freq + 1)) for word, freq in doc_freq.items()}


def process_tokens(doc_id, words, token_idf):
    # Обработка токенов
    tokens_path = os.path.join(TOKENS_DIR, f"tokens-{doc_id}.txt")
    with open(tokens_path, 'r', encoding='utf-8') as f:
        valid_tokens = [line.strip() for line in f if line.strip()]

    # Считаем TF для токенов
    token_counts = Counter(word for word in words if word in valid_tokens)
    total_token_words = sum(token_counts.values()) or 1
    tf_tokens = compute_tf(token_counts, total_token_words)

    # Сохраняем результаты
    token_output_path = os.path.join(OUTPUT_TOKENS, f"tfidf-{doc_id}.txt")
    with open(token_output_path, 'w', encoding='utf-8') as f:
        for token in valid_tokens:
            tf = tf_tokens.get(token, 0.0)
            idf = token_idf.get(token, 0.0)
            f.write(f"{token} {idf:.6f} {tf * idf:.6f}\n")


def process_lemmas(doc_id, words, lemma_idf):
    # Обработка лемм
    lemmas_path = os.path.join(LEMMAS_DIR, f"lemmas-{doc_id}.txt")
    lemma_forms = defaultdict(list)
    with open(lemmas_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lemma = parts[0]
                forms = parts[1:] if len(parts) > 1 else [lemma]
                lemma_forms[lemma] = forms

    # Считаем TF для лемм
    lemma_counts = Counter()
    for lemma, forms in lemma_forms.items():
        lemma_counts[lemma] += sum(words.count(form) for form in forms)

    total_lemma_words = sum(lemma_counts.values()) or 1
    tf_lemmas = compute_tf(lemma_counts, total_lemma_words)

    # Сохраняем результаты
    lemma_output_path = os.path.join(OUTPUT_LEMMAS, f"tfidf-{doc_id}.txt")
    with open(lemma_output_path, 'w', encoding='utf-8') as f:
        for lemma in lemma_forms:
            tf = tf_lemmas.get(lemma, 0.0)
            idf = lemma_idf.get(lemma, 0.0)
            f.write(f"{lemma} {idf:.6f} {tf * idf:.6f}\n")


def process_document(doc_id, doc_filename, token_idf, lemma_idf):
    html_path = os.path.join(PAGES_DIR, doc_filename)

    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_html = f.read()

    cleaned_text = normalize_text(preprocess_html(raw_html))

    # Токенизация: сначала разбиваем по пробелам, потом по дефисам
    words = []
    for word in cleaned_text.split():
        words.extend(word.split('-'))

    process_tokens(doc_id, words, token_idf)
    process_lemmas(doc_id, words, lemma_idf)


def main():
    doc_filenames = [f for f in os.listdir(PAGES_DIR) if f.endswith('.html')]
    doc_ids = [f.split('.')[0].split('_')[1] for f in doc_filenames]
    total_docs = len(doc_ids)

    # Считаем IDF
    token_doc_freq = Counter()
    lemma_doc_freq = Counter()

    # Собираем статистику по всем документам
    for doc_id in doc_ids:
        # Для токенов
        tokens_path = os.path.join(TOKENS_DIR, f"tokens-{doc_id}.txt")
        with open(tokens_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
            token_doc_freq.update(set(tokens))

        # Для лемм
        lemmas_path = os.path.join(LEMMAS_DIR, f"lemmas-{doc_id}.txt")
        with open(lemmas_path, 'r', encoding='utf-8') as f:
            for line in f:
                lemma = line.strip().split()[0]
                lemma_doc_freq[lemma] += 1

    token_idf = compute_idf(token_doc_freq, total_docs)
    lemma_idf = compute_idf(lemma_doc_freq, total_docs)

    # Обрабатываем все документы
    for doc_id, doc_filename in zip(doc_ids, doc_filenames):
        process_document(doc_id, doc_filename, token_idf, lemma_idf)

    print(f"- Токены: {OUTPUT_TOKENS}")
    print(f"- Леммы: {OUTPUT_LEMMAS}")


if __name__ == '__main__':
    main()
