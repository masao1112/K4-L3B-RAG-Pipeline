"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _convert_pdf_or_docx(path: Path) -> str:
    """Chuyển đổi file PDF/DOCX sang text/markdown với cơ chế fallback."""
    # 1. Thử dùng MarkItDown
    try:
        from markitdown import MarkItDown
        converter = MarkItDown()
        result = converter.convert(str(path))
        if result and result.text_content and len(result.text_content.strip()) >= 200:
            return result.text_content.strip()
    except Exception as error:
        print(f"MarkItDown notice on {path.name}: {error}")

    # 2. Fallback dùng pypdf
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"## Trang {i}\n\n{text.strip()}")
        full_text = "\n\n".join(pages)
        if len(full_text.strip()) >= 200:
            return full_text.strip()
    except Exception as error:
        print(f"pypdf notice on {path.name}: {error}")

    # 3. Fallback dùng fitz / PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc, 1):
            text = page.get_text() or ""
            if text.strip():
                pages.append(f"## Trang {i}\n\n{text.strip()}")
        full_text = "\n\n".join(pages)
        if len(full_text.strip()) >= 200:
            return full_text.strip()
    except Exception as error:
        print(f"fitz notice on {path.name}: {error}")

    raise RuntimeError(f"Không thể trích xuất nội dung từ: {path}")


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vào data/standardized/legal/."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() in {".pdf", ".doc", ".docx"} and not path.name.startswith("."):
            output_path = output_dir / f"{path.stem}.md"
            if output_path.exists() and len(output_path.read_text(encoding="utf-8").strip()) >= 200:
                print(f"Legal Markdown exists: {output_path}")
                continue
            text_content = _convert_pdf_or_docx(path)
            header = (
                f"# {path.stem.replace('-', ' ').title()}\n\n"
                f"**Source File:** `{path.name}`\n\n---\n\n"
            )
            output_path.write_text(header + text_content, encoding="utf-8")
            print(f"Converted legal: {output_path} ({len(text_content)} chars)")



def convert_news_articles() -> None:
    """Convert JSON vào data/standardized/news/."""
    import json
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(news_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        header = (
            f"# {data['title']}\n\n"
            f"**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        output_path = output_dir / f"{path.stem}.md"
        output_path.write_text(header + data["content_markdown"], encoding="utf-8")
        print(f"Converted news: {output_path}")



def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
