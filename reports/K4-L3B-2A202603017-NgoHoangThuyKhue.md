# Individual contribution report

Mỗi thành viên copy template này thành:

```text
reports/<student-id>-<short-name>.md
```

Giới hạn khuyến nghị: 1 trang, không chép lại README hoặc mô tả lý thuyết chung. Báo cáo không phải một bài pipeline cá nhân; mục đích là ghi nhận ownership và bằng chứng đóng góp trong sản phẩm nhóm.

---

## Thông tin

- Họ và tên: Ngô Hoàng Thụy Khuê
- Mã học viên: 2A202603017
- Nhóm: Bốn
- Repository/branch: khue

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Tìm và thu thập tài liệu pháp lý (Task 1) | Thêm code tải 3 PDF quy chế VinUni từ trang chính thức | `src/task1_collect_legal_docs.py` — commit `6e53ae5` | Done |
| Crawl tin tức (Task 2) | Thêm code crawl 5 bài viết học phí/tuyển sinh bằng Crawl4AI | `src/task2_crawl_news.py` - commit `6e53ae5` | Done |
| Chuẩn hóa Markdown (Task 3) | Thêm code chuyển PDF → Markdown (MarkItDown) và JSON → Markdown | `src/task3_convert_markdown.py` - commit `6e53ae5` | Done |

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chọn dữ liệu quy chế đào tạo VinUni (cử nhân, thạc sĩ, tiến sĩ) làm nguồn legal corpus.
   **Lý do/evidence:** Dữ liệu sạch, có cấu trúc heading rõ ràng, tải trực tiếp từ policy.vinuni.edu.vn.
   **Trade-off:** Ngôn ngữ không đồng nhất, quy chế cử nhân bằng tiếng Anh, thạc sĩ/tiến sĩ bằng tiếng Việt; điều này ảnh hưởng đến độ chính xác của embedding nhiều ngôn ngữ và có khả năng ảnh hưởng đến cách model trả lời.

2. **Quyết định:**  
   **Lý do/evidence:**  
   **Trade-off:**

## Kiểm thử và kết quả

- Test hoặc query tôi đã dùng: 
   - Query: "test query"
   - Test: `pytest tests/test_contracts.py -q`, `pytest tests/test_acceptance.py -q` 
- Kết quả trước/sau nếu có:
   - `pytest tests/test_contracts.py -q` 15 passed, `pytest tests/test_acceptance.py -q` 5 passed
- Lỗi đã phát hiện và cách xử lý:
  - Một số URL không crawl được trong lần đầu vì lỗi access/timeout → crawl lại những web này. 

## Điều còn hạn chế

- Chưa kiểm tra kỹ tài liệu tải về: một số PDF bị thiếu vài trang do timeout mạng, cần cơ chế retry và xác thực kích thước file.
- Nếu có thêm thời gian, tôi sẽ thêm testcase khó, injection prompt, để đánh giá độ robust của pipeline.

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/09/2026
- Tên thành viên: Ngô Hoàng Thụy Khuê
