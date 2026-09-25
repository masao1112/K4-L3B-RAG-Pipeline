# Danh Sách Thành Viên & Phân Công Nhiệm Vụ

- **Chủ đề dự án:** Hệ thống RAG Hỏi đáp Tuyển sinh, Học phí & Quy chế Đào tạo - Trường Đại học VinUniversity
- **Repository:** `masao1112/K4-L3B-RAG-Pipeline`

---

## Bảng phân công thành viên

| STT | Họ và tên | Mã học viên | Vai trò (Role) | Nhánh Git (Branch) | Phần việc đảm nhiệm |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 1 | **Lương Nguyễn Tiến Anh** | 2A202603002 | Data Engineer & Chunking Lead | `TienAnh` | - Thu thập & chuẩn hóa tài liệu (Task 1, 2, 3)<br>- Thiết kế chiến lược Chunking & Embedding (Task 4) |
| 2 | **Đặng Quang Huy** | 2A202602962 | Pipeline & Retrieval Engineer | `quanghuy` | - Recursive chunking & Vector DB Chroma (Task 4)<br>- Semantic Search, BM25 & RRF Reranking (Task 5, 6, 7)<br>- Fallback & Retrieval Pipeline (Task 8, 9)<br>- Generation có Citation & Safe Refusal (Task 10) |
| 3 | **Ngô Hoàng Thụy Khuê** | 2A202603017 | Frontend Engineer & Data Curator | `khue` | - Thu thập & làm giàu dữ liệu tin tức tuyển sinh (Task 2, 3)<br>- Phát triển giao diện Chatbot Streamlit (`app.py`)<br>- Hiển thị trích dẫn nguồn, điểm tương đồng & tham số |
| 4 | **Đào Quang Thái Anh** | 2A202602987 | QA & Evaluation Lead | `thaianh/02987` | - Kiểm định chất lượng dữ liệu và hợp đồng module (`tests/`)<br>- Xây dựng Golden Dataset $\ge 15$ câu (`golden_dataset.json`)<br>- Đánh giá 4 chỉ số RAG (A/B Testing) & Hoàn thiện `RESULT.md` |

---

## Chi tiết nhiệm vụ theo Module

### 1. Dữ liệu & Chuẩn hóa (Data & Preprocessing)
- **Phụ trách:** Lương Nguyễn Tiến Anh, Đào Quang Thái Anh, Ngô Hoàng Thủy Khuê
- **Nội dung:** 
  - [src/task1_collect_legal_docs.py](src/task1_collect_legal_docs.py): 3 văn bản quy chế đào tạo PDF (Đại học, Thạc sĩ, Tiến sĩ).
  - [src/task2_crawl_news.py](src/task2_crawl_news.py): 5 bài viết thông tin tuyển sinh, học phí dạng JSON.
  - [src/task3_convert_markdown.py](src/task3_convert_markdown.py): Chuyển đổi và chuẩn hóa sang Markdown trong `data/standardized/`.

### 2. Chunking, Indexing & Retrieval (RAG Core)
- **Phụ trách:** Đặng Quang Huy, Lương Nguyễn Tiến Anh
- **Nội dung:**
  - [src/task4_chunking_indexing.py](src/task4_chunking_indexing.py): Recursive Character Text Splitter, embedding với BAAI/bge-m3, index ChromaDB.
  - [src/task5_semantic_search.py](src/task5_semantic_search.py): Dense search với Cosine similarity.
  - [src/task6_lexical_search.py](src/task6_lexical_search.py): Lexical search với BM25Okapi.
  - [src/task7_reranking.py](src/task7_reranking.py): Thuật toán Reciprocal Rank Fusion (RRF $k=60$).
  - [src/task8_pageindex_vectorless.py](src/task8_pageindex_vectorless.py) & [src/task9_retrieval_pipeline.py](src/task9_retrieval_pipeline.py): Pipeline hợp nhất, kiểm tra threshold và cơ chế fallback.

### 3. Generation & Giao diện (Generation & UI)
- **Phụ trách:** Đặng Quang Huy, Ngô Hoàng Thủy Khuê
- **Nội dung:**
  - [src/task10_generation.py](src/task10_generation.py): Reordering chống lost-in-the-middle, sinh câu trả lời kèm citation và safe refusal.
  - [app.py](app.py): Giao diện Streamlit tương tác mượt mà, hiển thị rõ nguồn tham khảo và thông số truy xuất.

### 4. Đánh giá & Báo cáo (Evaluation & Reports)
- **Phụ trách:** Đào Quang Thái Anh
- **Nội dung:**
  - [group_project/evaluation/golden_dataset.json](group_project/evaluation/golden_dataset.json): Bộ 15 câu hỏi - đáp chuẩn grounded.
  - [group_project/evaluation/RESULT.md](group_project/evaluation/RESULT.md): Báo cáo đo lường 4 metrics (Faithfulness, Relevance, Recall, Precision) và thực nghiệm A/B Testing giữa Config A (Dense-only) và Config B (Hybrid + RRF).
  - Báo cáo đóng góp cá nhân theo mẫu [reports/INDIVIDUAL_REPORT.md](reports/INDIVIDUAL_REPORT.md).