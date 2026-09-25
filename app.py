"""Streamlit interface for the VinUni grounded RAG assistant."""

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation
from src.task4_chunking_indexing import get_collection


load_dotenv()

st.set_page_config(
    page_title="VinUni Knowledge Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


SAMPLE_QUESTIONS = (
    "Khoản hỗ trợ học phí cử nhân VinUni là bao nhiêu phần trăm và ai tài trợ?",
    "Chương trình MBA tại VinUni kéo dài bao lâu và học phí là bao nhiêu?",
    "Điều kiện ngoại ngữ đầu vào chương trình tiến sĩ là gì?",
    "Quy trình ứng tuyển đại học VinUni gồm những vòng nào?",
)

METHOD_LABELS = {
    "dense": "Dense retrieval",
    "bm25": "BM25",
    "hybrid": "Hybrid · RRF",
    "pageindex": "PageIndex fallback",
    "none": "Không có nguồn",
}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #102638;
            --muted: #64737d;
            --paper: #f7f5ef;
            --surface: #fffdf8;
            --line: #dedbd2;
            --navy: #102638;
            --coral: #df6254;
            --teal: #1c7568;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 8%, rgba(223, 98, 84, 0.08), transparent 24rem),
                var(--paper);
            color: var(--ink);
        }

        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stMainBlockContainer"] {
            max-width: 1120px;
            padding-top: 3rem;
            padding-bottom: 8rem;
        }

        [data-testid="stSidebar"] {
            background: var(--navy);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        [data-testid="stSidebar"] * { color: #edf2f3; }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] .stMarkdown p { color: #b7c3c8; }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.1);
        }

        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.16);
            background: rgba(255, 255, 255, 0.05);
            color: #f7fafb;
        }

        [data-testid="stSidebar"] .stButton button:hover {
            border-color: rgba(255, 255, 255, 0.32);
            background: rgba(255, 255, 255, 0.09);
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            margin: 4px 0 24px;
        }

        .brand-mark {
            display: grid;
            place-items: center;
            width: 36px;
            height: 36px;
            border: 1px solid rgba(255, 255, 255, 0.22);
            color: #ffb7a8;
            font-size: 18px;
            line-height: 1;
        }

        .brand-name {
            color: #ffffff;
            font-weight: 720;
            letter-spacing: -0.02em;
        }

        .brand-subtitle {
            color: #9fb0b8;
            font-size: 11px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .sidebar-label {
            color: #9fb0b8;
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 20px 0 8px;
        }

        .system-status {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            font-size: 13px;
        }

        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-right: 7px;
            border-radius: 50%;
            background: #54c9a8;
            box-shadow: 0 0 0 4px rgba(84, 201, 168, 0.12);
        }

        .status-value {
            color: #dce7e9;
            font-variant-numeric: tabular-nums;
            font-size: 12px;
        }

        .hero {
            border-bottom: 1px solid var(--line);
            margin-bottom: 28px;
            padding-bottom: 28px;
        }

        .eyebrow {
            color: var(--teal);
            font-size: 11px;
            font-weight: 760;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }

        .hero h1 {
            color: var(--ink);
            font-size: clamp(2rem, 5vw, 3.5rem);
            font-weight: 740;
            letter-spacing: -0.045em;
            line-height: 1.02;
            margin: 0 0 16px;
            max-width: 760px;
        }

        .hero p {
            color: var(--muted);
            font-size: 16px;
            line-height: 1.65;
            margin: 0;
            max-width: 720px;
        }

        .hero-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 20px;
        }

        .hero-chip {
            border: 1px solid var(--line);
            background: rgba(255, 253, 248, 0.72);
            color: #4d606b;
            font-size: 12px;
            padding: 6px 10px;
            border-radius: 999px;
        }

        [data-testid="stChatMessage"] {
            background: rgba(255, 253, 248, 0.86);
            border: 1px solid var(--line);
            border-radius: 14px;
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
            margin-bottom: 14px;
            padding: 8px 12px;
        }

        [data-testid="stChatMessage"] p {
            color: var(--ink);
            line-height: 1.68;
        }

        [data-testid="stChatInput"] {
            background: rgba(247, 245, 239, 0.92);
            border-top: 1px solid rgba(222, 219, 210, 0.9);
            backdrop-filter: blur(14px);
            padding-top: 12px;
        }

        [data-testid="stChatInput"] textarea { color: var(--ink); }

        [data-testid="stChatInput"] > div {
            border-color: #c9c6bd;
            background: var(--surface);
            border-radius: 12px;
        }

        [data-testid="stChatInput"] > div:focus-within {
            border-color: var(--teal);
            box-shadow: 0 0 0 3px rgba(28, 117, 104, 0.12);
        }

        .section-kicker {
            color: var(--muted);
            font-size: 11px;
            font-weight: 720;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 16px 0 10px;
        }

        .method-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #126458;
            background: rgba(28, 117, 104, 0.09);
            border: 1px solid rgba(28, 117, 104, 0.18);
            border-radius: 999px;
            font-size: 11px;
            font-weight: 650;
            padding: 4px 8px;
            margin: 4px 0 8px;
        }

        [data-testid="stExpander"] {
            background: rgba(255, 253, 248, 0.64);
            border: 1px solid var(--line);
            border-radius: 10px;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary:hover { color: var(--teal); }

        .source-meta {
            color: var(--muted);
            font-size: 12px;
            font-variant-numeric: tabular-nums;
            margin-bottom: 10px;
        }

        .empty-state {
            background: rgba(255, 253, 248, 0.56);
            border: 1px dashed #cbc7bd;
            border-radius: 14px;
            color: var(--muted);
            margin: 8px 0 20px;
            padding: 18px 20px;
        }

        .empty-state strong {
            color: var(--ink);
            display: block;
            margin-bottom: 4px;
        }

        .stButton button {
            border-color: #cfcbc2;
            background: rgba(255, 253, 248, 0.7);
            color: var(--ink);
            min-height: 44px;
            transition: border-color 150ms ease, background 150ms ease, transform 150ms ease;
        }

        .stButton button:hover {
            border-color: var(--teal);
            background: rgba(28, 117, 104, 0.06);
            color: #125f55;
        }

        .stButton button:active { transform: scale(0.985); }

        @media (max-width: 720px) {
            [data-testid="stMainBlockContainer"] { padding-top: 1.5rem; }
            .hero h1 { font-size: 2.2rem; }
            .hero p { font-size: 14px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=30, show_spinner=False)
def _collection_size() -> int:
    try:
        return int(get_collection().count())
    except Exception:
        return 0


def _render_source(source: dict[str, Any], index: int) -> None:
    metadata = source.get("metadata") or {}
    title = metadata.get("title") or metadata.get("source") or "Tài liệu không tên"
    source_name = metadata.get("source") or "Không rõ nguồn"
    method = METHOD_LABELS.get(source.get("retrieval_method"), "Retrieval")
    score = float(source.get("score", 0.0))
    chunk_index = metadata.get("chunk_index", 0)

    with st.expander(f"{index}. {title}"):
        st.markdown(
            f'<div class="source-meta">{method} · score {score:.4f} · '
            f'{source_name} · chunk {chunk_index}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(source.get("content") or "Không có nội dung trích dẫn.")
        url = metadata.get("url")
        if isinstance(url, str) and url.strip():
            st.link_button("Mở nguồn gốc ↗", url, use_container_width=False)


def _render_assistant_message(message: dict[str, Any]) -> None:
    st.markdown(message["content"])
    sources = message.get("sources") or []
    retrieval_source = message.get("retrieval_source", "none")
    if sources:
        label = METHOD_LABELS.get(retrieval_source, retrieval_source)
        st.markdown(
            f'<div class="method-pill">● {label} · {len(sources)} nguồn</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-kicker">Nguồn kiểm chứng</div>', unsafe_allow_html=True)
        for index, source in enumerate(sources, start=1):
            _render_source(source, index)


def _render_sidebar() -> None:
    provider = os.getenv("LLM_PROVIDER", "chưa cấu hình").strip().lower()
    model = os.getenv("LLM_MODEL", "").strip() or "chưa cấu hình"
    chunk_count = _collection_size()

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">◈</div>
                <div>
                    <div class="brand-name">VinUni Assistant</div>
                    <div class="brand-subtitle">Grounded knowledge</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Trạng thái hệ thống</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="system-status">
                <span><span class="status-dot"></span>Vector index</span>
                <span class="status-value">{chunk_count} chunks</span>
            </div>
            <div class="system-status">
                <span><span class="status-dot"></span>LLM provider</span>
                <span class="status-value">{provider}</span>
            </div>
            <div class="system-status">
                <span>Model</span>
                <span class="status-value">{model}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-label">Thiết lập truy xuất</div>', unsafe_allow_html=True)
        st.caption("Số đoạn tài liệu tối đa dùng để tạo câu trả lời.")
        st.slider(
            "Số nguồn",
            min_value=3,
            max_value=10,
            key="top_k",
            label_visibility="collapsed",
        )

        st.markdown("---")
        if st.button("Xóa cuộc trò chuyện", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.caption(
            "Câu trả lời được tạo từ 3 quy chế và 5 bài viết VinUni. "
            "Hãy mở phần nguồn để kiểm chứng thông tin."
        )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">VinUni knowledge base · 8 verified documents</div>
            <h1>Tra cứu quy chế.<br>Nhận câu trả lời có nguồn.</h1>
            <p>
                Hỏi về học phí, tuyển sinh và quy chế đào tạo VinUni. Mỗi câu trả lời
                được tổng hợp từ kho tài liệu nội bộ và đi kèm đoạn trích để đối chiếu.
            </p>
            <div class="hero-meta">
                <span class="hero-chip">Hybrid search</span>
                <span class="hero-chip">Reciprocal Rank Fusion</span>
                <span class="hero-chip">Citation-first</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_suggestions() -> str | None:
    st.markdown(
        """
        <div class="empty-state">
            <strong>Bắt đầu với một câu hỏi cụ thể</strong>
            Chọn gợi ý bên dưới hoặc nhập câu hỏi của bạn vào ô chat.
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected: str | None = None
    columns = st.columns(2)
    for index, question in enumerate(SAMPLE_QUESTIONS):
        with columns[index % 2]:
            if st.button(question, key=f"sample-{index}", use_container_width=True):
                selected = question
    return selected


def _answer_query(query: str, top_k: int) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=":material/school:"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.status("Đang tìm bằng chứng trong kho tài liệu…", expanded=False) as status:
            try:
                result = generate_with_citation(query, top_k=top_k)
                status.update(label="Đã tổng hợp câu trả lời", state="complete")
            except Exception as exc:
                status.update(label="Không thể hoàn tất truy vấn", state="error")
                result = {
                    "answer": "Hệ thống gặp lỗi khi xử lý câu hỏi. Vui lòng thử lại.",
                    "sources": [],
                    "retrieval_source": "none",
                }
                st.caption(f"Chi tiết kỹ thuật: {type(exc).__name__}")

        message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": result.get("sources", []),
            "retrieval_source": result.get("retrieval_source", "none"),
        }
        _render_assistant_message(message)
        st.session_state.messages.append(message)


def main() -> None:
    _inject_styles()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = 5

    _render_sidebar()
    _render_hero()

    for message in st.session_state.messages:
        avatar = (
            ":material/school:"
            if message["role"] == "user"
            else ":material/auto_awesome:"
        )
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant":
                _render_assistant_message(message)
            else:
                st.markdown(message["content"])

    suggested_query = None
    if not st.session_state.messages:
        suggested_query = _render_suggestions()

    typed_query = st.chat_input("Ví dụ: Học phí chương trình MBA là bao nhiêu?")
    query = suggested_query or typed_query
    if query:
        _answer_query(query.strip(), int(st.session_state.top_k))


if __name__ == "__main__":
    main()
