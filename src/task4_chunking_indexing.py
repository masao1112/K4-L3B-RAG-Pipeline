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

from pathlib import Path


import re


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# Các phương pháp hỗ trợ: "fixed" | "token" | "markdown" | "semantic" | "recursive"
CHUNKING_METHOD = "fixed"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


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
def split_semantic(text: str, max_chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Cắt dựa trên ranh giới câu/ngữ nghĩa."""
    if not text.strip():
        return []
    raw_sentences = re.split(r"(?<=[.?!;:\n])\s+", text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return split_fixed(text, max_chunk_size, chunk_overlap)

    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        if len(sent) > int(max_chunk_size * 1.1):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            chunks.extend(split_fixed(sent, max_chunk_size, chunk_overlap))
            continue

        if current_len + len(sent) + 1 > max_chunk_size:
            if current_chunk:
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
        # Fallback pure-python recursive splitting
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    # TODO: Dispatch theo EMBEDDING_PROVIDER trong .env.
    #
    # Provider local gợi ý:
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # return model.encode(texts).tolist()
    raise NotImplementedError("Implement embed_texts")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    # TODO: Tạo hoặc mở persistent collection.
    #
    # import chromadb
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # return client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    raise NotImplementedError("Implement get_collection")


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if path.name.startswith("."):
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        documents.append({
            "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
            "content": content,
            "metadata": {
                "source": path.name,
                "title": path.stem,
                "doc_type": doc_type,
                "url": None,
            },
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    chunks = []
    for document in documents:
        doc_id = document["id"]
        doc_meta = document["metadata"]
        text_chunks = split_text(document["content"], method=CHUNKING_METHOD)
        for index, text in enumerate(text_chunks):
            if not text.strip():
                continue
            chunks.append({
                "id": f"{doc_id}::chunk-{index}",
                "content": text,
                "metadata": {**doc_meta, "chunk_index": index},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    # TODO: Embed theo batch và giữ nguyên các field của chunk.
    #
    # vectors = embed_texts([chunk["content"] for chunk in chunks])
    # for chunk, vector in zip(chunks, vectors):
    #     chunk["embedding"] = vector
    # return chunks
    raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    # TODO: Upsert ids, documents, embeddings và metadatas.
    #
    # collection = get_collection()
    # collection.upsert(
    #     ids=[chunk["id"] for chunk in chunks],
    #     documents=[chunk["content"] for chunk in chunks],
    #     embeddings=[chunk["embedding"] for chunk in chunks],
    #     metadatas=[chunk["metadata"] for chunk in chunks],
    # )
    raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
