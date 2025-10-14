# =============================================
# 1. Импорты
# =============================================
from __future__ import annotations

import json
import os
import uuid
from textwrap import shorten
from typing import List, Dict, Any
import requests
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from yandex_cloud_ml_sdk import YCloudML
import textwrap

# =============================================
# 2. Настройки
# =============================================

# Путь к папке с документами
DOCS_PATH = "Task2/knowledge_base"  # папка с .txt файлами
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_PROMPT_CHARS = 6000  # ограничение на длину контекста в символах (адаптируй под модель)
MAX_FRAGMENT_PREVIEW = 1200  # сколько символов показывать из каждого чанка в prompt

# Загрузка модели эмбеддингов
model = SentenceTransformer(EMBEDDING_MODEL)

FEW_SHOT_EXAMPLES = [
    {
        "question": "Расскажи про локацию Туманная Буря",
        "answer": (
            "Тип локации: Аномальная атмосферная зона"
            "Класс опасности: Средний–Высокий"
            "Редкость: Редкий"
            "Регион: Восточные Холмы"
            "Так как класс опасности средне-высокий нужно подобрать хорошую экипировку"
            "Файл: Локация_Туманная_Буря.txt"
        ),
    },
    {
        "question": "Какое оружие можно найти в сером периметре?",
        "answer": (
            "Можно найти оружие Тень Пустоши, которая имеет тип Снайперская система с ПСИ-фильтром"
            "Файл: Оружие_Тень_Пустоши.txt"
        ),
    },
]

# =============================================
# 3. Функция для чтения текстовых файлов
# =============================================
def load_documents(path: str):
    docs = []
    for fname in os.listdir(path):
        if fname.endswith(".txt"):
            with open(os.path.join(path, fname), "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({
                "path": os.path.join(path, fname),
                "title": fname,
                "content": text
            })
    return docs


# =============================================
# 4. Разбиение на логические чанки
# =============================================
def chunk_document(doc, chunk_size=1500, chunk_overlap=0):
    """
    Разбивает текст на чанки с сохранением метаданных.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(doc["content"])

    tags = doc["title"].split("_")

    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": str(uuid.uuid4()),
            "text": chunk,
            "metadata": {
                "source": doc["path"],
                "title": doc["title"],
                "chunk_index": i,
                "tags": tags
            }
        })
    return chunk_data


# =============================================
# 5. Генерация эмбеддингов
# =============================================
def embed_chunks(chunks):
    texts = [ch["text"] for ch in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


# =============================================
# 6. Создание FAISS-индекса
# =============================================
def build_faiss_index(chunks, embeddings):
    d = embeddings.shape[1]  # размерность эмбеддингов
    index = faiss.IndexFlatL2(d)
    index.train(embeddings)
    index.add(embeddings)
    return index

# =============================================
# 8. Поиск по запросу
# =============================================
def search(query, index, chunks, k=3):
    """
    Ищет наиболее похожие чанки для заданного запроса.
    """
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    print(f"\n🔎 Результаты для запроса: '{query}'\n")

    print(I)  # индексы соседей
    print(D)  # косинусные близости

    # for i, idx in enumerate(I[0]):
    #     chunk = chunks[idx]
    #     print(f"Результат #{i + 1}")
    #     print(f"Источник: {chunk['metadata']['title']}")
    #     print(f"Файл: {chunk['metadata']['source']}")
    #     print(f"Текст чанка:\n{chunk['text'][:400]}...\n")
    #     print("=" * 80)

    return D, I


def build_prompt(query: str, hits_idx: List[int], metadata: List[Dict[str, Any]]) -> str:
    """
    Собираем промпт: системная инструкция -> найденные фрагменты (с ссылками на источник) -> запрос пользователя.
    Ограничиваем суммарную длину контекста MAX_PROMPT_CHARS.

    header = (
        "Ты — помощник, использующий базу знаний. "
        "Ниже — выдержки из документов с указанием источников. "
        "Ответь на вопрос пользователя, основываясь на этих фрагментах. Если не знаешь ответа так и скажи.\n\n"
    )
    """

    few_shot_text = "Примеры правильных ответов:\n\n"
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_text += f"Вопрос: {ex['question']}\nОтвет: {ex['answer']}\n---\n"

    context_parts = []
    used_chars = 0
    for idx in hits_idx:
        try:
            chunk_meta = metadata[idx]
        except IndexError:
            continue
        text = chunk_meta.get("text") or chunk_meta.get("content") or ""
        source = chunk_meta.get("metadata", {}).get("source", chunk_meta.get("source", "<unknown>"))
        title = chunk_meta.get("metadata", {}).get("title", chunk_meta.get("title", ""))
        chunk_index = chunk_meta.get("metadata", {}).get("chunk_index", chunk_meta.get("chunk_index", None))
        preview = shorten(text, width=MAX_FRAGMENT_PREVIEW, placeholder=" ...")
        fragment = f"Источник: {title} | Файл: {source} | Чанк: {chunk_index}\n{preview}\n---\n"
        if used_chars + len(fragment) > MAX_PROMPT_CHARS:
            break
        context_parts.append(fragment)
        used_chars += len(fragment)

    context = "".join(context_parts)
    user_section = f"Вопрос: {query}\n\nОтвет (кратко и с указанием источников, если возможно):"

    prompt = few_shot_text + context + user_section
    return prompt

# =============================================
# 7. Основной конвейер
# =============================================
def main():
    # Загружаем документы
    documents = load_documents(DOCS_PATH)
    all_chunks = []

    # Разбиваем каждый документ на чанки
    for doc in documents:
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)

    # Генерируем эмбеддинги
    embeddings = embed_chunks(all_chunks)

    # Создаем FAISS-индекс
    index = build_faiss_index(all_chunks, embeddings)

    # Сохраняем метаданные
    with open("metadata.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Сохраняем индекс
    faiss.write_index(index, "faiss.index")
    print("✅ Индекс и метаданные успешно созданы.")

    # Пример поиска
    query = "Где можно найти снайперскую систему?"
    D, I = search(query, index, all_chunks)

    promt = build_prompt(query, I[0], all_chunks)

    sdk = YCloudML(
        folder_id=os.getenv('folder_id'), auth=os.getenv('auth')
    )

    model = sdk.models.completions("yandexgpt", model_version="rc")
    model = model.configure(temperature=0.3)
    result = model.run(
        [
            {
                "role": "system",
                "text": "Ты — помощник, использующий базу знаний, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги"
                        "Пример твоего размышления"
                        "1. Сначала найду, какая технология используется в HyperRelay."
                        "2. В документе указано, что HyperRelay питается от ядра VoidCore."
                        "3. Следовательно, ответ — VoidCore."
                        "Ниже — выдержки из документов с указанием источников. "
                        "Ответь на вопрос пользователя, основываясь на этих фрагментах. Если не знаешь ответа так и скажи.\n\n",
            },
            {
                "role": "user",
                "text": promt,
            },
        ]
    )

    wrapper = textwrap.TextWrapper(initial_indent="* ", width=100)

    for alternative in result:
        new_text =  wrapper.fill(str(alternative))
        print(new_text)

# =============================================
# 9. Запуск
# =============================================
if __name__ == "__main__":
    main()
