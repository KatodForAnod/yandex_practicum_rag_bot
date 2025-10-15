# =============================================
# 1. Импорты
# =============================================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import uuid
import json

# =============================================
# 2. Настройки
# =============================================

# Путь к папке с документами
DOCS_PATH = "Task2/knowledge_base"  # папка с .txt файлами
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Загрузка модели эмбеддингов
model = SentenceTransformer(EMBEDDING_MODEL)

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
    index.add(embeddings)
    return index

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
    query = "класс опасности экипировка"
    search(query, index, all_chunks)

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

    print(I)          # индексы соседей
    print(D)    # косинусные близости

    for i, idx in enumerate(I[0]):
        chunk = chunks[idx]
        print(f"Результат #{i+1}")
        print(f"Источник: {chunk['metadata']['title']}")
        print(f"Файл: {chunk['metadata']['source']}")
        print(f"Текст чанка:\n{chunk['text'][:400]}...\n")
        print("="*80)

# =============================================
# 9. Запуск
# =============================================
if __name__ == "__main__":
    main()
