# RAG evaluation results

## Run information

| Field | Value |
|---|---|
| Evaluation date | 25/09/2026 |
| Framework and version | `custom-tfidf-evaluator-v1`; `ragas==0.4.3` được cài đặt nhưng không dùng làm judge trong run này |
| Evaluator | TF-IDF word unigram/bigram cosine, relevance threshold `0.20` |
| Generator model | `gemini-3.5-flash-lite`, temperature `0.3`, top-p `0.9` |
| Embedding model | `BAAI/bge-m3`, vector 1024 chiều |
| Corpus version/commit | `33de451`; Chroma collection `rag_documents`, 70 chunks |
| Golden dataset size | 17 câu hỏi grounded |
| `top_k` | 5; lấy 10 candidates trước khi cắt top-5 |
| Fallback threshold and calibration | Pipeline dùng dense cosine threshold `0.30`; PageIndex được tắt trong A/B để chỉ thay retrieval strategy. Threshold metric `0.20` được chọn sau khi kiểm tra phân bố similarity của 170 cặp reference–chunk. |

Kết quả chi tiết cho từng câu, source ID, latency và lỗi provider được lưu trong [`metrics.json`](metrics.json). Có thể chạy lại bằng:

```bash
python group_project/evaluation/run_evaluation.py --workers 2
```

## Configurations

- **Config A — dense-only:** embed câu hỏi bằng BGE-M3, truy vấn Chroma cosine lấy 10 candidates rồi giữ 5 kết quả có score cao nhất.
- **Config B — hybrid + RRF:** dùng cùng 10 dense candidates, thêm 10 BM25 candidates trên cùng corpus, hợp nhất bằng Reciprocal Rank Fusion với `rrf_k=60`, rồi giữ top-5.

Hai cấu hình dùng cùng 17 câu hỏi, corpus, generator, prompt, `top_k` và evaluator. PageIndex fallback không tham gia để tránh tạo thêm biến ngoài retrieval strategy.

### Metric definitions

- **Faithfulness:** trung bình cosine lớn nhất giữa mỗi claim trong câu trả lời và các chunks được truy xuất.
- **Answer relevance:** trung bình cosine answer–question và answer–expected-answer.
- **Context recall:** tỷ lệ claim trong expected context có ít nhất một chunk đạt similarity từ `0.20`.
- **Context precision:** Average Precision@5, trong đó chunk được xem là liên quan khi cosine với expected context đạt từ `0.20`.

Đây là evaluator tự động, xác định và tái lập được; các điểm số là proxy nội bộ để so sánh A/B, không được diễn giải như điểm RAGAS do LLM chấm.

## Overall scores

| Metric | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Faithfulness | 0.2841 | **0.3480** | **+0.0639** |
| Answer relevance | 0.3822 | **0.3878** | **+0.0056** |
| Context recall | 0.5007 | **0.5056** | **+0.0049** |
| Context precision | 0.5281 | **0.5343** | **+0.0062** |
| **Average** | **0.4238** | **0.4439** | **+0.0201** |

| Runtime | Config A | Config B | Delta B−A |
|---|---:|---:|---:|
| Mean retrieval latency | 2.2037 s | 2.2109 s | +0.0072 s |
| Mean generation latency | 2.876 s | 2.879 s | +0.003 s |
| Provider errors in final run | 0/17 | 0/17 | 0 |

## A/B comparison

- **Cấu hình tốt hơn:** Config B — hybrid + RRF, tăng average `+0.0201` điểm tuyệt đối (khoảng `+4.7%` tương đối) so với dense-only.
- **Evidence:** cải thiện lớn nhất nằm ở faithfulness (`+0.0639`). Hybrid khắc phục rõ ba câu mà dense thiếu evidence: A-ACC (`+0.3262` average/case), điểm đạt học phần thạc sĩ (`+0.2242`) và thời gian đào tạo tiến sĩ (`+0.3933`). BM25 bổ sung các chunks chứa acronym, mốc điểm và con số chính xác.
- **Trade-off về latency/cost:** hybrid thêm BM25 và RRF nhưng retrieval chỉ tăng trung bình `0.0072 s` trong run này vì query embedding BGE-M3 vẫn là chi phí chính. Lượng context gửi cho Gemini giữ nguyên top-5 nên generation latency và token budget gần như không đổi.
- **Cảnh báo:** hybrid không thắng ở mọi câu. Ba regression lớn là Course Withdrawal (`−0.3238`), CGPA (`−0.2782`) và quy trình ứng tuyển (`−0.1123`), do RRF đẩy chunk dense chứa bằng chứng ra khỏi top-5.

## Worst performers

| # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage | Root cause |
|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | Trụ cột A-ACC gồm những yếu tố nào? | Dense-only | 0.1480 | 0.1753 | 0.0000 | 0.0000 | Retrieval | Dense chỉ lấy chunk giới thiệu A-ACC/EXCEL, không lấy chunk kế tiếp chứa bốn thành phần; generator từ chối đúng vì thiếu evidence. |
| 2 | Thời hạn Course Withdrawal và giới hạn tín chỉ? | Hybrid + RRF | 0.0835 | 0.3063 | 0.0000 | 0.0000 | Fusion/retrieval | Chunk cử nhân chứa mốc 30% và 18 tín chỉ có trong dense top-5 nhưng bị RRF thay bằng chunks thạc sĩ/tin tức ít liên quan. |
| 3 | Ngưỡng CGPA Xuất sắc và Giỏi? | Hybrid + RRF | 0.0512 | 0.3507 | 0.0000 | 0.0000 | Fusion/retrieval | Chunk chứa bảng CGPA (`vinuni-policy-cu-nhan::chunk-16`) xuất hiện ở dense nhưng bị loại sau fusion; LLM chuyển sang safe refusal. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
|---:|---|---|---|---|
| 1 | Chunk theo cấu trúc heading/bảng và thêm section title vào metadata/content trước khi embed | Bằng chứng A-ACC, CGPA và Course Withdrawal nằm ở chunk liền kề hoặc bảng tách khỏi tiêu đề | Tăng recall cho acronym, bảng điểm và điều khoản pháp lý | Re-index rồi chạy lại 17 cases; mục tiêu context recall ≥ 0.60 và ba worst cases > 0 |
| 2 | Hiệu chỉnh fusion: bảo toàn dense top-1/top-2 hoặc dùng weighted RRF theo query | Hybrid làm mất đúng chunk ở cases 6, 12 và 13 dù dense đã tìm thấy | Giữ lợi ích BM25 nhưng giảm regression do fusion | So sánh RRF hiện tại với protected-dense RRF trên cùng evaluator; không chấp nhận case giảm quá 0.10 |
| 3 | Thử query expansion/reranker cho truy vấn song ngữ và acronym | Hybrid cải thiện mạnh A-ACC và các câu chứa mốc số, nhưng còn nhạy với cách diễn đạt Việt–Anh | Cải thiện precision và thứ hạng chunk đúng | Thêm experiment riêng, báo cáo delta metric và p95 latency trên 17 cases |

## Bonus experiments

Chưa chạy HyDE hoặc cross-encoder reranker trong lần đánh giá này. Kết quả hybrid + RRF ở trên là A/B bắt buộc của pipeline, không được tính là bonus. Thử nghiệm ưu tiên tiếp theo là protected-dense weighted RRF; chỉ đề xuất nhận bonus nếu có run độc lập chứng minh metric tăng và báo cáo latency.

## Limitations

- Golden dataset chỉ có 17 câu in-domain và chưa có tập out-of-domain để đo trực tiếp safe-refusal rate.
- TF-IDF evaluator minh bạch và không tốn judge API nhưng đánh giá thấp các paraphrase có ít từ trùng; cần thêm human review hoặc LLM-as-judge độc lập trước khi diễn giải điểm tuyệt đối.
- Latency được đo trên máy CPU hiện tại; lượt nạp BGE-M3 đầu tiên không tính vào mean per-query sau warm-up.
