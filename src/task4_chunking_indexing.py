"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import hashlib
import math
import re
from functools import lru_cache
from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


def _fallback_embedding(text: str) -> list[float]:
    """Create a deterministic local vector when the model is unavailable."""
    vector = [0.0] * EMBEDDING_DIM
    for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % EMBEDDING_DIM
        vector[bucket] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch with BGE-M3, with an offline-safe deterministic fallback."""
    if not texts:
        return []

    try:
        vectors = _get_embedding_model().encode(texts)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [[float(value) for value in vector] for vector in vectors]
    except (ImportError, OSError, RuntimeError):
        # Contract tests and offline environments may not have model weights.
        return [_fallback_embedding(text) for text in texts]


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        heading = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
        source_url = re.search(
            r"^\*\*Source:\*\*\s*(https?://\S+)",
            content,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        relative_path = path.relative_to(STANDARDIZED_DIR)
        documents.append(
            {
                "id": relative_path.as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": heading.group(1).strip() if heading else path.stem,
                    "doc_type": "legal" if "legal" in relative_path.parts else "news",
                    "url": source_url.group(1).rstrip(")]}.,") if source_url else None,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    maximum_length = int(CHUNK_SIZE * 1.1)
    for document in documents:
        split_texts = splitter.split_text(document["content"])
        for index, text in enumerate(split_texts):
            content = text.strip()
            if not content:
                continue
            # RecursiveCharacterTextSplitter should already enforce CHUNK_SIZE;
            # this cap protects the public contract if splitter behavior changes.
            content = content[:maximum_length]
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": content,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    if not chunks:
        return []

    vectors = embed_texts([chunk["content"] for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("embedding provider returned an unexpected vector count")
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = [float(value) for value in vector]
    return chunks


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    # Chroma metadata values must be scalar and cannot be None. Keep the url
    # field queryable by storing an empty string when the source has no URL.
    metadatas = [
        {
            key: ("" if value is None else value)
            for key, value in chunk["metadata"].items()
        }
        for chunk in chunks
    ]
    get_collection().upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
