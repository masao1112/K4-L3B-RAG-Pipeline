"""
Task 4 — Chunking, embedding và indexing.

Quy trình:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng phương pháp đã chọn (hỗ trợ: semantic, markdown, token, fixed, recursive).
    3. Tạo embedding cho từng chunk theo provider cấu hình trong .env.
    4. Upsert vào ChromaDB với cosine distance.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = os.getenv("CHUNKING_METHOD", "semantic")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024
COLLECTION_NAME = "rag_documents"

_MODEL_CACHE = {}


def _get_sentence_transformer_model(model_name: str):
    """Cache model SentenceTransformer để tránh nạp lại nhiều lần."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Tạo vector embeddings theo EMBEDDING_PROVIDER trong .env."""
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    if provider in ("sentence_transformers", "local", "hf"):
        model = _get_sentence_transformer_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        target_model = model_name if model_name and "/" not in model_name else "text-embedding-3-small"
        response = client.embeddings.create(input=texts, model=target_model)
        return [item.embedding for item in response.data]

    elif provider in ("gemini", "google"):
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        target_model = model_name if model_name and "/" not in model_name else "text-embedding-004"
        result = client.models.embed_content(model=target_model, contents=texts)
        return [item.values for item in result.embeddings]

    else:
        model = _get_sentence_transformer_model(model_name)
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# 1. FIXED / CHARACTER CHUNKING
def split_fixed(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt cố định theo số lượng ký tự với overlap."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    text_len = len(text)
    step = max(1, chunk_size - chunk_overlap)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_len:
            break
        start += step
    return chunks


# 2. TOKEN-BASED CHUNKING
def split_token(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt theo số lượng token bằng tiktoken hoặc từ."""
    if not text.strip():
        return []
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        max_tokens = max(10, int(chunk_size / 4.2))
        overlap_tokens = max(2, int(chunk_overlap / 4.2))
        step = max(1, max_tokens - overlap_tokens)
        chunks = []
        for i in range(0, len(tokens), step):
            tok_slice = tokens[i : i + max_tokens]
            decoded = enc.decode(tok_slice).strip()
            if len(decoded) > int(chunk_size * 1.1):
                decoded = decoded[: int(chunk_size * 1.1)].strip()
            if decoded:
                chunks.append(decoded)
            if i + max_tokens >= len(tokens):
                break
        return chunks
    except Exception:
        words = text.split()
        max_words = max(5, int(chunk_size / 6.0))
        overlap_words = max(1, int(chunk_overlap / 6.0))
        step = max(1, max_words - overlap_words)
        chunks = []
        for i in range(0, len(words), step):
            slice_words = words[i : i + max_words]
            chunk = " ".join(slice_words).strip()
            if len(chunk) > int(chunk_size * 1.1):
                chunk = chunk[: int(chunk_size * 1.1)].strip()
            if chunk:
                chunks.append(chunk)
            if i + max_words >= len(words):
                break
        return chunks


# 3. MARKDOWN STRUCTURE-AWARE CHUNKING
def split_markdown(text: str, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt dựa trên cấu trúc tiêu đề Markdown (#, ##, ###, ####)."""
    if not text.strip():
        return []
    lines = text.split("\n")
    sections = []
    current_lines = []

    for line in lines:
        if line.strip().startswith(("# ", "## ", "### ", "#### ")):
            if current_lines:
                sec = "\n".join(current_lines).strip()
                if sec:
                    sections.append(sec)
                current_lines = []
        current_lines.append(line)

    if current_lines:
        sec = "\n".join(current_lines).strip()
        if sec:
            sections.append(sec)

    if not sections:
        return split_fixed(text, max_chunk_size, chunk_overlap)

    chunks = []
    for section in sections:
        if len(section) <= int(max_chunk_size * 1.1):
            chunks.append(section)
        else:
            sub = split_fixed(section, max_chunk_size, chunk_overlap)
            chunks.extend(sub)
    return chunks


# 4. SEMANTIC / SENTENCE-LEVEL CHUNKING
def compute_sentence_similarities(sentences: list[str]) -> list[float]:
    """Tính độ tương đồng cosine giữa các câu liên tiếp bằng Embedding hoặc TF-IDF Vectorizer."""
    if len(sentences) <= 1:
        return []

    if _MODEL_CACHE:
        try:
            import numpy as np
            unique_sentences = list(dict.fromkeys(sentences))
            unique_embs = embed_texts(unique_sentences)
            emb_map = {s: v for s, v in zip(unique_sentences, unique_embs)}
            sims = []
            for i in range(len(sentences) - 1):
                v1 = np.array(emb_map[sentences[i]])
                v2 = np.array(emb_map[sentences[i + 1]])
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                sim = float(np.dot(v1, v2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 1.0
                sims.append(sim)
            return sims
        except Exception:
            pass

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(sentences)
        sims = [float(cosine_similarity(vec[i], vec[i + 1])[0][0]) for i in range(len(sentences) - 1)]
        return sims
    except Exception:
        return []


def split_semantic(
    text: str,
    max_chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    similarity_threshold: float = 0.3,
) -> list[str]:
    """
    Semantic chunking:
    1. Tách văn bản thành các câu hoàn chỉnh.
    2. Đo độ tương đồng ngữ nghĩa cosine giữa các câu liền kề.
    3. Xác định điểm ngắt khi độ tương đồng giảm mạnh (topic shift) hoặc khi vượt max_chunk_size.
    """
    if not text.strip():
        return []

    raw_sentences = re.split(r"(?<=[.?!;:\n])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return split_fixed(text, max_chunk_size, chunk_overlap)
    if len(sentences) == 1:
        if len(sentences[0]) > int(max_chunk_size * 1.1):
            return split_fixed(sentences[0], max_chunk_size, chunk_overlap)
        return sentences

    similarities = compute_sentence_similarities(sentences)
    threshold = similarity_threshold
    if similarities:
        import numpy as np
        p30 = float(np.percentile(similarities, 30))
        threshold = max(0.05, min(similarity_threshold, p30))

    chunks = []
    current_chunk = []
    current_len = 0

    for i, sent in enumerate(sentences):
        if len(sent) > int(max_chunk_size * 1.1):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            chunks.extend(split_fixed(sent, max_chunk_size, chunk_overlap))
            continue

        is_topic_shift = False
        if similarities and i > 0 and (i - 1) < len(similarities):
            if similarities[i - 1] <= threshold:
                is_topic_shift = True

        if current_chunk and (current_len + len(sent) + 1 > max_chunk_size or is_topic_shift):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(sent)
        current_len += len(sent) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# 5. RECURSIVE CHUNKING
def split_recursive(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt đệ quy theo các ký tự phân tách phổ biến."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except Exception:
        return split_semantic(text, max_chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_text(text: str, method: str = CHUNKING_METHOD) -> list[str]:
    """Hàm điều phối phân tách văn bản theo method đã chọn."""
    method_normalized = method.lower()
    if method_normalized in ("fixed", "character"):
        return split_fixed(text)
    elif method_normalized == "token":
        return split_token(text)
    elif method_normalized in ("markdown", "header"):
        return split_markdown(text)
    elif method_normalized == "semantic":
        return split_semantic(text)
    elif method_normalized == "recursive":
        return split_recursive(text)
    else:
        raise ValueError(f"Unknown chunking method: {method}. Supported: fixed, token, markdown, semantic, recursive")


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
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
    chunks: list[dict] = []
    maximum_length = int(CHUNK_SIZE * 1.1)
    for document in documents:
        split_texts = split_text(document["content"], method=CHUNKING_METHOD)
        for index, text in enumerate(split_texts):
            content = text.strip()
            if not content:
                continue
            content = content[:maximum_length]
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": content,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


def embed_chunks(chunks: list[dict], batch_size: int = 32) -> list[dict]:
    """Thêm embedding vào từng chunk theo batch."""
    if not chunks:
        return []

    contents = [chunk["content"] for chunk in chunks]
    all_embeddings = []
    for i in range(0, len(contents), batch_size):
        batch = contents[i : i + batch_size]
        all_embeddings.extend(embed_texts(batch))

    if len(all_embeddings) != len(chunks):
        raise ValueError("embedding provider returned an unexpected vector count")

    for chunk, vector in zip(chunks, all_embeddings):
        chunk["embedding"] = [float(value) for value in vector]
    return chunks


def index_to_vectorstore(chunks: list[dict], batch_size: int = 100) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return

    collection = get_collection()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        metadatas = [
            {
                key: ("" if value is None else value)
                for key, value in chunk["metadata"].items()
            }
            for chunk in batch
        ]
        collection.upsert(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["content"] for chunk in batch],
            embeddings=[chunk["embedding"] for chunk in batch],
            metadatas=metadatas,
        )


def run_pipeline() -> None:
    """Chạy toàn bộ quy trình: load, chunk, embed và index."""
    print("1. Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents.")

    print(f"2. Chunking with method '{CHUNKING_METHOD}'...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("3. Generating embeddings...")
    embedded_chunks = embed_chunks(chunks)
    print(f"Generated embeddings for {len(embedded_chunks)} chunks.")

    print("4. Indexing into ChromaDB...")
    index_to_vectorstore(embedded_chunks)
    print(f"Successfully indexed {len(embedded_chunks)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    run_pipeline()
