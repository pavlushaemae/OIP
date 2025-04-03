import os
import re
import html
import nltk
import pymorphy3
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import defaultdict


def preprocess_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    for tag in soup(['script', 'style', 'meta', 'link', 'noscript']):
        tag.extract()
    return soup.get_text(separator=' ', strip=True)


def normalize_text(text):
    text = html.unescape(text)
    text = re.sub(r'[^а-яА-ЯёЁ\s-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()


def find_valid_tokens(cleaned_text, russian_stopwords):
    # Токенизация с учетом дефисов
    words = re.findall(r'\b(?:[а-яё]+-)*[а-яё]+\b', cleaned_text)

    morph = pymorphy3.MorphAnalyzer()
    valid_tokens = set()

    for word in words:
        # Разбиваем слово на части по дефисам
        parts = word.split('-')

        for part in parts:
            if len(part) < 3:
                continue

            # Проверяем валидность каждой части отдельно
            parsed = morph.parse(part)
            if not parsed:
                continue

            valid_pos = {'NOUN', 'ADJF', 'ADJS', 'VERB', 'INFN', 'PRTF', 'PRTS', 'GRND'}
            valid = any(
                p.score > 0.3 and
                p.tag.POS in valid_pos and
                len(p.normal_form) > 2
                for p in parsed
            )

            if valid and part not in russian_stopwords:
                valid_tokens.add(part)

    return valid_tokens


def main():
    nltk.download("stopwords")
    russian_stopwords = set(stopwords.words("russian")).union({'т.д.', 'др.', 'т.п.'})
    directory = 'input/'
    processed_ids = {}
    morph = pymorphy3.MorphAnalyzer()

    for filename in sorted(os.listdir(directory)):
        if not filename.endswith('.html'):
            continue

        try:
            doc_id = int(filename.split('.')[0].split('_')[1])
        except (ValueError, IndexError):
            print(f"Пропущен файл с некорректным ID: {filename}")
            continue

        if doc_id in processed_ids:
            print(f"Обнаружен дубликат ID {doc_id} в файле {filename}")
            continue

        file_path = os.path.join(directory, filename)

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_html = f.read()

        cleaned_text = normalize_text(preprocess_html(raw_html))

        # Сортируем токены
        filtered_tokens = sorted(find_valid_tokens(cleaned_text, russian_stopwords))

        # Формируем леммы
        lemmas = defaultdict(set)
        for token in filtered_tokens:
            parsed = morph.parse(token)[0]
            lemma = parsed.normal_form
            lemmas[lemma].add(token)

        # Сохраняем результаты
        with open(f'output/tokens/tokens-{doc_id}.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(filtered_tokens))

        with open(f'output/lemmas/lemmas-{doc_id}.txt', 'w', encoding='utf-8') as f:
            for lemma in sorted(lemmas):
                # Всегда ставим нормальную форму первой
                forms = sorted(lemmas[lemma])
                if lemma not in forms:
                    forms = [lemma] + forms
                else:
                    # Если лемма уже есть, перемещаем её в начало
                    forms.remove(lemma)
                    forms.insert(0, lemma)
                f.write(f"{' '.join(forms)}\n")


    print("Обработка завершена. Файлы сохранены.")


if __name__ == "__main__":
    main()