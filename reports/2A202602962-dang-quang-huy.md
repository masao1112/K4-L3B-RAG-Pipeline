# Báo cáo đóng góp cá nhân

## Thông tin

- **Họ và tên:** Đặng Quang Huy
- **Mã học viên:** 2A202602962
- **Nhóm:** K4-L3B — RAG Pipeline VinUniversity
- **Repository/branch:** `masao1112/K4-L3B-RAG-Pipeline` / `DangQuangHuy-02962`

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Chunking, embedding và vector database (Task 4) | Hoàn thiện recursive chunking, ID chunk ổn định, metadata nguồn, embedding BGE-M3, cơ chế fallback cục bộ và upsert ChromaDB dùng cosine distance. Tạo index gồm 70 chunks để ứng dụng có thể chạy ngay. | `src/task4_chunking_indexing.py`, `chroma_db/`; commits `f1b2461`, `7a2176c` | Done |
| Hybrid retrieval (Task 5–7) | Cài đặt dense semantic search, BM25Okapi và Reciprocal Rank Fusion; chuẩn hóa kết quả theo cùng `SearchResult` contract và giới hạn đúng `top_k`. | `src/task5_semantic_search.py`, `src/task6_lexical_search.py`, `src/task7_reranking.py`; commit `f1b2461` | Done |
| Fallback và retrieval pipeline (Task 8–9) | Tích hợp PageIndex fallback; dùng cosine score gốc của dense retrieval để so threshold, chỉ fusion RRF một lần và trả hybrid results nếu fallback lỗi. | `src/task8_pageindex_vectorless.py`, `src/task9_retrieval_pipeline.py`; commit `f1b2461` | Done |
| Generation có citation (Task 10) | Xây dựng context có title/source, reorder giảm lost-in-the-middle, hỗ trợ OpenAI/Gemini/Anthropic, citation `[Document N]` và safe refusal khi thiếu evidence hoặc provider lỗi. | `src/task10_generation.py`; commit `f1b2461` | Done |
| Chatbot Streamlit end-to-end | Thay UI placeholder bằng giao diện hoàn chỉnh: lịch sử chat, câu hỏi gợi ý, điều chỉnh `top_k`, trạng thái hệ thống, retrieval method/score và danh sách nguồn mở rộng được. Thêm cấu hình Streamlit và xử lý lỗi UI. | `app.py`, `.streamlit/config.toml`; commit `7a2176c` | Done |
| Kiểm thử và tích hợp | Chạy contract tests, kiểm thử trình duyệt bằng Playwright, xác minh truy vấn thật với Gemini và merge nhánh cá nhân lên `main`, ưu tiên phiên bản pipeline của nhánh khi conflict. | `tests/test_contracts.py`; merge commit `c9ba7d2` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Kết hợp dense retrieval và BM25 bằng RRF; chỉ dùng cosine score gốc để quyết định PageIndex fallback.  
   **Lý do/evidence:** Dense retrieval bắt được tương đồng ngữ nghĩa, còn BM25 giữ lợi thế với tên chương trình, con số và thuật ngữ chính xác. RRF hợp nhất thứ hạng mà không so trực tiếp hai thang điểm khác nhau. Contract tests xác nhận pipeline chỉ fusion một lần và vẫn hoạt động khi fallback provider lỗi.  
   **Trade-off:** Mỗi câu hỏi phải chạy hai bộ truy xuất nên tốn thời gian hơn dense-only; RRF score chỉ biểu diễn thứ hạng, không phải xác suất liên quan.

2. **Quyết định:** Thiết kế generation theo hướng citation-first và safe refusal, đồng thời hiển thị đầy đủ evidence trên UI.  
   **Lý do/evidence:** Context đánh số từng document cùng title/source giúp câu trả lời map ngược được về chunk gốc. Truy vấn kiểm thử về hỗ trợ học phí trả đúng “35%, do Tập đoàn Vingroup tài trợ” và kèm 5 nguồn kiểm chứng.  
   **Trade-off:** Prompt và context dài hơn, tăng latency/token; khi provider hoặc retrieval lỗi, hệ thống ưu tiên từ chối thay vì cố tạo câu trả lời.

## Kiểm thử và kết quả

- **Test hoặc query đã dùng:** `pytest tests/test_contracts.py -q`; kiểm tra syntax bằng `python -m py_compile`; kiểm thử UI thật bằng Playwright; truy vấn “Khoản hỗ trợ học phí cử nhân VinUni là bao nhiêu phần trăm và ai tài trợ?”.
- **Kết quả:** 15/15 contract tests pass; ChromaDB index thành công 70 chunks; giao diện trả câu trả lời grounded đúng 35% và Vingroup, hiển thị retrieval method cùng 5 nguồn; Streamlit health endpoint trả HTTP 200.
- **Lỗi đã phát hiện và cách xử lý:**
  - Gemini `gemini-2.5-flash-lite` trả 404 do model không còn cấp cho người dùng mới; cập nhật cấu hình chạy sang `gemini-3.5-flash-lite` và gọi thử API thành công.
  - Streamlit hiểu ký hiệu avatar dạng text là đường dẫn ảnh; thay bằng Material icons hợp lệ.
  - File watcher quét namespace của Transformers gây nhiều lỗi phụ thuộc `torchvision`; tắt watcher trong cấu hình Streamlit vì ứng dụng chỉ dùng pipeline văn bản.

## Điều còn hạn chế

- **Hạn chế cụ thể:** Lượt truy vấn đầu tiên trên máy chỉ có CPU còn chậm do phải nạp BGE-M3; PageIndex fallback cũng cần API key riêng mới hoạt động đầy đủ.
- **Nếu có thêm thời gian:** Tôi sẽ thêm bước warm-up embedding khi khởi động, đo latency theo từng stage và hiệu chỉnh threshold/top-k trên golden dataset thay vì chỉ dùng cấu hình mặc định.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- **Ngày:** 25/09/2026
- **Tên thành viên:** Đặng Quang Huy
