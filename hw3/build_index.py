import os
import re
import html
import nltk
import json
import pymorphy3
from bs4 import BeautifulSoup
from nltk.corpus import stopwords


def normalize_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^а-яА-ЯёЁ\s-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def find_valid_tokens(cleaned_text, russian_stopwords):
    # Токенизация с учетом дефисов
    morph = pymorphy3.MorphAnalyzer()
    words = re.findall(r'\b(?:[а-яё]+-)*[а-яё]+\b', cleaned_text)

    lemmas = set()
    for word in words:
        for part in word.split('-'):
            part = part.strip()
            if len(part) < 3 or part in russian_stopwords:
                continue

            parsed = morph.parse(part)
            if not parsed:
                continue

            valid_pos = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'}
            best_parse = max(parsed, key=lambda x: x.score)

            if best_parse.score > 0.3 and best_parse.tag.POS in valid_pos:
                lemma = best_parse.normal_form
                if len(lemma) > 2:
                    lemmas.add(lemma)

    return lemmas


def process_document(file_path, russian_stopwords):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')

        for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        cleaned_text = normalize_text(text)
        return find_valid_tokens(cleaned_text, russian_stopwords)


def build_inverted_index(directory, russian_stopwords):
    inverted_index = {}
    doc_ids = {}

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.html'):
            continue

        try:
            doc_id = int(filename.split('.')[0].split('_')[1])
        except (ValueError, IndexError):
            print(f"Пропущен файл с некорректным ID: {filename}")
            continue

        if doc_id in doc_ids:
            print(f"Обнаружен дубликат ID {doc_id} в файле {filename}")
            continue

        file_path = os.path.join(directory, filename)
        doc_ids[doc_id] = filename

        # Начальные формы на данном файле
        doc_lemmas = process_document(file_path, russian_stopwords)
        for lemma in doc_lemmas:
            inverted_index.setdefault(lemma, []).append(doc_id)

    inverted_index = {k: sorted(list(set(v))) for k, v in inverted_index.items()}
    return inverted_index, doc_ids


if __name__ == "__main__":
    nltk.download('stopwords')
    russian_stopwords = set(stopwords.words('russian'))
    custom_stopwords = {'т.д.', 'др.', 'т.п.'}
    russian_stopwords.update(custom_stopwords)

    inverted_index, doc_ids = build_inverted_index('input/', russian_stopwords)

    with open('inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump({
            'inverted_index': inverted_index,
            'documents': doc_ids
        }, f, ensure_ascii=False, indent=2)

    print("Инвертированный индекс создан")
