import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Выбираем категорию Википедии
WIKI_CATEGORY_URL = "https://ru.wikipedia.org/wiki/Категория:История_математики"
BASE_WIKI_URL = "https://ru.wikipedia.org"
SAVE_DIR = "output"
INDEX_FILE = os.path.join(SAVE_DIR, "index.txt")

os.makedirs(SAVE_DIR, exist_ok=True)

# Получаем список страниц
response = requests.get(WIKI_CATEGORY_URL)
soup = BeautifulSoup(response.text, "html.parser")

URLS = []
for link in soup.select(".mw-category-group a"):
    href = link.get("href")
    if href and href.startswith("/wiki/") and not href.endswith(('.js', '.css')):
        URLS.append(BASE_WIKI_URL + href)

# Ограничиваем список до 100 ссылок
URLS = URLS[:100]

print(f"Найдено {len(URLS)} страниц.")

index_entries = []

for i, url in enumerate(tqdm(URLS, desc="Downloading"), start=1):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Очистка страницы от <script> и <link rel="stylesheet">
        page_soup = BeautifulSoup(response.text, "html.parser")
        for script in page_soup("script"):
            script.decompose()
        for css_link in page_soup.find_all("link", rel="stylesheet"):
            css_link.decompose()

        filename = f"page_{i}.html"
        file_path = os.path.join(SAVE_DIR, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(page_soup))

        index_entries.append(f"{i} {url}\n")
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")

with open(INDEX_FILE, "w", encoding="utf-8") as f:
    f.writelines(index_entries)

print("Download completed. Check the 'output' folder.")
