# Báo cáo Day 7

Học viên: Cao Phạm Thành Đạt — MSSV 2A202603970. Ngày chạy: 16/11/2026. Code và hướng dẫn chạy: [src/README.md](../../src/README.md).

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang và số chunk: corpus Day 6 (repo `K5-L3B-Day06-CaoPhamThanhDat-2A202603970-DataIngestion`, commit `9aab448`): *The Art of War* (Sun Tzu, bản dịch Lionel Giles 1910, Project Gutenberg #132) và *The Prince* (Machiavelli, bản dịch W. K. Marriott, #1232). Public domain tại Hoa Kỳ ([manifest](../../data/day7/day6_manifest.json)). 204 "trang logic" (đơn vị ≥1500 ký tự theo chương, không phải trang in) → 476 chunk theo chiến lược B của Day 6 (gộp đoạn, tối đa 1200 / tối thiểu 200 ký tự). ID chunk dạng `AOW-p056-B001` = sách-trang-chunk.
- Vector store, embedding model, phiên bản và số chiều: Chroma 1.5.9 (persistent, HNSW cosine, collection `day6_B`), `BAAI/bge-small-en-v1.5` qua fastembed 0.9.1, 384 chiều; document dùng `embed`, query dùng `query_embed` như Day 6. Thư mục `chroma_db` Day 6 không được commit nên index được dựng lại bằng cùng code chunking, model và ID ([prepare_data.py](../../src/prepare_data.py), 74 giây, chi phí 0). Kiểm chứng: dense top-3 trên dev cho Hit@1 7/16, Hit@3 8/16 — **trùng khớp** kết quả Day 6 chiến lược B.
- Agent model, cách chạy và phân chia dev/test: `openai/gpt-4o-mini` qua OpenRouter, temperature 0, tool calling (`search`, `answer`, `abstain`), vòng lặp `starter/agentic.py::run_agent`. Benchmark 26 câu ([questions.json](../../data/day7/questions.json), [cases.json](../../data/day7/cases.json)): dev 16 có đáp án + 2 thiếu bằng chứng; test 6 có đáp án + 2 thiếu bằng chứng. 22 câu có đáp án lấy từ Day 6 (split đóng băng từ Day 6, chương test tách khỏi dev; Day 6 chọn cấu hình bằng dev nên test chưa dùng để chỉnh). 4 câu thiếu bằng chứng (u01–u04) do tôi thêm cho Day 7.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm: câu có đáp án có `gold` = {source_id, page, evidence} với evidence nằm trọn trong một chunk (kiểm tra bằng code khi tạo dữ liệu). Câu thiếu bằng chứng gần chủ đề, kiểm tra bằng grep **toàn bộ** 204 đơn vị (không dựa vào top-k), ví dụ u01 "Lorenzo"/"dedicat" 0 kết quả vì lời đề tặng nằm trước CHAPTER I và đã bị loại khi ingest; u03 "gunpowder/firearm/cannon" 0 kết quả trong AOW. Hit = đúng nguồn + trang và chứa nguyên văn evidence (`starter/baseline.py::hit`). Gold chỉ dùng để chấm sau khi chạy; agent chỉ nhận câu hỏi và `filter` do người dùng cung cấp (q02, q16, q20, q21: `{"book": ...}` vì câu hỏi tự nêu tên sách).

## Retrieval và thử nghiệm

Bốn phương pháp ([common.py](../../src/common.py)): **dense** (Chroma Day 6); **BM25** (Okapi k1=1.5, b=0.75, bỏ stopword tiếng Anh); **hybrid** = RRF (k=60) trên top-20 dense + top-20 BM25 (hợp nhất theo rank, không cộng điểm thô); **rerank** = hybrid top-20 xếp lại bằng cross-encoder `Xenova/ms-marco-MiniLM-L-6-v2`. Bảng dưới là query gốc, không agent ([results/retrieval](../../results/retrieval/)); context = `pack_context` trên top-k với budget tính bằng **ký tự** (gồm header nguồn/trang). Chi phí dịch vụ = 0 (chạy local); thời gian = trung vị ms/query, không gồm thời gian load model.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (R1) | dense, k=3, filter người dùng, không rerank, 4000 ký tự | Dense Day 6 | dev | 7/16 (43.8%) | 8/16 (50.0%) | 8/16 (50.0%) | 26 ms / $0 |
| Phương pháp 2 (R2) | BM25, k=3, 4000 | Khớp từ chính xác | dev | 5/16 (31.3%) | 6/16 (37.5%) | 6/16 (37.5%) | 0.7 ms / $0 |
| Phương pháp 3 (R3) | hybrid RRF, k=3, 4000 | Kết hợp hai tín hiệu | dev | 6/16 (37.5%) | 8/16 (50.0%) | 8/16 (50.0%) | 23 ms / $0 |
| Phương pháp 4 (R4) | hybrid → rerank cross-encoder, k=3, 4000 | Xếp lại 20 candidate | dev | 7/16 (43.8%) | 10/16 (62.5%) | 10/16 (62.5%) | 780 ms / $0 |
| Cải thiện 1 (R5, R6) | dense k=5 / hybrid k=5, 6000 | Tăng top-k và budget | dev | 7/16; 6/16 | 8/16; 8/16 | 8/16; 9/16 (56.3%) | 22 / 20 ms |
| Cải thiện 2 (R7) | rerank k=5, 6000 (R8: 9000 cho cùng kết quả) | Rerank + top-k lớn hơn | dev | 7/16 (43.8%) | 10/16 (62.5%) | 11/16 (68.8%) | 795 ms / $0 |
| Cấu hình đã chọn (R7/R8) | rerank k=5, budget 9000 (retrieval của agent A6) | Chốt trước test | test | 5/6 (83.3%) | 5/6 (83.3%) | 5/6 (83.3%) | 764 ms / $0 |
| Baseline (R1) | dense k=3, 4000 | | test | 4/6 (66.7%) | 4/6 (66.7%) | 4/6 (66.7%) | 33 ms / $0 |

MRR@10 dev: dense 0.496, BM25 0.387, hybrid 0.458, rerank 0.547. BM25 tự nó yếu nhất; hybrid không hơn dense ở k=3 nhưng cung cấp candidate tốt hơn cho reranker (rerank đưa q03, q04 vào top-3 và q16 vào top-10). Một số câu trượt top-3 ở mọi phương pháp với query gốc (q06, q07, q11): từ trong câu hỏi khác hẳn bản dịch cổ ("occupying army" ↔ "colonies").

**Hai lỗi dev, nguyên nhân, cách thử (agent)**

1. *Lỗi 1 — từ chối nhầm dù evidence đã có trong context* (q10, q13, q14 ở A1: gold chunk rank 1). Trace cho thấy model gọi `answer` với `citations: ["PRN-p014-B001"]` nhưng **không ghi `[PRN-p014-B001]` trong câu trả lời**, nên `run_agent` coi citation không hợp lệ và trả abstain. Thử: prompt v2 bắt buộc citation inline, có ví dụ (A4, cùng retriever dense k3 để tách riêng ảnh hưởng của prompt). Kết quả dev: từ chối nhầm 10/16 → 6/16; grounded trung bình 0.438 → 1.188. Q13/q14 thành 2 điểm.
2. *Lỗi 2 — tìm lại nhưng lặp gần nguyên query* (q11, q15, q16 ở A1–A3: 3 lần search gần giống nhau rồi abstain). Thử: v2 yêu cầu đổi từ vựng theo cách bản dịch cổ diễn đạt + rerank k5, budget 9000 để chunk từ lần search sau vẫn vừa context (A5). Đọc trace A5 phát hiện thêm: v2 khuyên "nêu tên sách", khiến query chứa "Machiavelli" và luôn trả về cùng nhóm chunk nhắc tên tác giả (q09 từ 2 điểm ở A4 tụt về abstain). v3 bỏ chỉ dẫn đó và cấm tên tác giả/sách trong query (A6): q09, q16 đúng (q16 tìm thấy PRN-p047-B001 ở lần search thứ 3). q11, q15 vẫn sai (thiếu từ "colonies", "arms/mercenaries") — giới hạn còn lại.

Lý do chọn: tiêu chí công bố trước trong [configs/agent_dev.json](../../configs/agent_dev.json) — (1) grounded answer trung bình dev, (2) abstention đúng, (3) ít từ chối nhầm, (4) độ trễ/chi phí. A6 cao nhất ở (1) và (3), bằng ở (2), nên được chốt ([configs/agent_test.json](../../configs/agent_test.json)) trước khi chạy test. Không chỉnh gì sau khi xem test.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi: 3 tool — `search(query)` (retriever theo cấu hình, filter lấy từ input người dùng), `answer(answer, citations)`, `abstain(reason)`; `tool_choice=required`. Tối đa 3 search/câu (cho phép 2 lần viết lại query; chưa thử giới hạn khác), dừng khi model trả lời/từ chối, hoặc đạt giới hạn → abstain. Mỗi lượt model nhận câu hỏi, danh sách query đã dùng và context tích lũy mới nhất (không nhân budget). Context: `pack_context` của starter — khử trùng ID, giữ thứ tự chunk từ search trước, bỏ chunk không vừa budget; mỗi block có `[id | nguồn | page]`. Câu trả lời không có citation hợp lệ trong context → abstain. Nội dung tài liệu bọc trong `<context>` và prompt ghi rõ là dữ liệu, không phải chỉ dẫn. Lỗi model/mạng: retry 3 lần rồi ghi `status: "error"` (không có lần lỗi nào; không có fallback giả).

| Cấu hình / split | Tool use | Số lượt gọi (search: số câu) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A1 dense k3, prompt v1 / dev | 18/18 | 1:13, 3:5 | 12/1/3 — 0.44 | 4/6 | 6/6 | 2/2 | 10/16 |
| A2 hybrid k3, v1 / dev | 18/18 | 1:13, 2:1, 3:4 | 11/2/3 — 0.50 | 5/7 | 7/7 | 2/2 | 9/16 |
| A3 rerank k5, v1 / dev | 18/18 | 1:13, 3:5 | 9/1/6 — 0.81 | 8/10 | 9/9 | 2/2 | 7/16 |
| A4 dense k3, v2 / dev (cải thiện 1) | 18/18 | 1:9, 2:1, 3:8 | 6/1/9 — 1.19 | 12/12 | 10/10 | 2/2 | 6/16 |
| A5 rerank k5 b9000, v2 / dev (cải thiện 2) | 18/18 | 1:11, 2:1, 3:6 | 7/1/8 — 1.06 | 14/16 | 11/11 | 2/2 | 5/16 |
| **A6 rerank k5 b9000, v3 / dev (chọn)** | 18/18 | 1:11, 2:1, 3:6 | 5/0/11 — **1.38** | 17/19 | 12/12 | 2/2 | 4/16 |
| A1 baseline / test | 8/8 | 1:5, 2:1, 3:2 | 3/0/3 — 1.00 | 3/4 | 4/4 | 2/2 | 2/6 |
| **A6 đã chọn / test** | 8/8 | 1:6, 3:2 | 0/1/5 — **1.83** | 6/7 | 6/6 | 2/2 | 0/6 |

Context Evidence Rate của agent (evidence gold trong context cuối): dev A1 8/16, A2 7/16, A3 9/16, A4 8/16, A5 9/16, A6 11/16; test A1 5/6, A6 6/6. Chấm context thủ công 0/1/2 trên 6 câu test có đáp án: A1 — mức 2: 5 câu, mức 0: 1 câu (q18, context chỉ có đoạn sĩ quan nổi giận); A6 — mức 2: 6/6 (đủ evidence, block có nguồn và trang). Lưu ý A1 có evidence trong context ở q17, q19 nhưng vẫn từ chối (lỗi 1).

Thời gian/chi phí agent (trung vị giây/câu; tổng token; tổng USD theo OpenRouter `usage.cost`): dev A1 4.8 s, 36.8k, $0.0064; A2 4.4 s, 35.5k, $0.0061; A3 5.2 s, 50.1k, $0.0084; A4 5.5 s, 53.8k, $0.0089; A5 5.3 s, 64.7k, $0.0105; A6 5.5 s, 69.0k, $0.0113. Test A1 5.5 s, 16.3k, $0.0028; A6 5.3 s, 27.0k, $0.0044. Tổng toàn bộ thí nghiệm ≈ $0.059. A6 tốn ~1.9× token so với A1 (context 5 chunk, budget 9000) và thêm ~0.8 s rerank/search trên CPU.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q17 (A6, test) | có đáp án, điều kiện | 1 search "precaution when crossing salt marshes" → AOW-p056-B001 rank 1 | "get over them quickly, without any delay" [AOW-p056-B001] | 2 — đúng nguyên văn gold, citation hỗ trợ |
| q18 (A6, test) | có đáp án | 1 search "a ruler may initiate conflict out of wrath" → AOW-p106-B001 | Không được ra quân chỉ để "gratify his own spleen"/"pique" [AOW-p106-B001] | 2 — gold §18 trong chunk |
| q19 (A6, test) | có đáp án | 1 search "who are used as local informants or spies" → AOW-p110-B001 | Dân địa phương ("inhabitants of a district") [AOW-p110-B001] | 2 — gold |
| q20 (A6, test) | có đáp án, filter the_prince | 1 search → PRN-p055-B002 | Cáo để phát hiện bẫy, sư tử để dọa sói [PRN-p055-B002] | 2 — gold |
| q21 (A6, test) | có đáp án, filter the_prince | 1 search "best fortress for a ruler" → PRN-p072-B002 | Không bị dân ghét [PRN-p072-B002] | 2 — gold |
| q22 (A6, test) | có đáp án | 1 search → PRN-p078-B001 + 4 chunk khác | Cho phép nói thật nhưng "respect for you abates" [PRN-p078-B001]; câu cuối [PRN-p060-B003] | 1 — ý chính đúng; câu cuối trích chunk về âm mưu, không hỗ trợ (citation sai 1/2) |
| u03 (A6, test) | thiếu evidence | 3 search: "gunpowder firearms siege warfare", "gunpowder artillery…", "use of gunpowder…" | Từ chối | đúng — không khẳng định vô căn cứ |
| u04 (A6, test) | thiếu evidence | 3 search về khẩu phần gạo/ngày | Từ chối | đúng — corpus chỉ có đơn vị picul, không có khẩu phần |
| q18 (A1, test) — **lỗi** | có đáp án | 2 search "start a war out of anger" → AOW-p071-B001 (sĩ quan nổi giận) | "Không nên khởi chiến vì giận" [AOW-p071-B001] | 0 — kết luận đúng nhưng chunk không nói về vua; không có evidence |
| q13 (A1, dev) — **lỗi** | có đáp án | 1 search → PRN-p014-B001 rank 1 | Model gửi citation nhưng không ghi `[id]` trong câu → bị chuyển thành abstain | 0 — từ chối nhầm; sửa ở A4/A6 |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí: trace từng câu (tool call, tham số, query, kết quả id/rank/score, context cuối, câu trả lời, citation, từng lần gọi LLM với token/chi phí/độ trễ) ở `results/agent/<split>/<config>/traces.json`; nhãn ở `reviews.json` (kèm lý do), tổng hợp `agent_metrics.json` (từ `starter/agent_metrics.py`), `run_summary.json`. Retrieval: `results/retrieval/<split>/`. Nhãn được viết trong [write_reviews.py](../../src/write_reviews.py). Lệnh chạy: [src/README.md](../../src/README.md). Unit test: `python -m unittest discover -s tests -v` → 17 test OK (15 starter + 2 mới cho BM25/RRF, không gọi model).
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn: trên dev, A1 → A6 tăng grounded 0.44 → 1.38/2 và giảm từ chối nhầm 10 → 4/16; trên test (sau khi chốt) 1.00 → 1.83/2, từ chối nhầm 2 → 0/6, abstention đúng giữ 2/2 ở cả hai. Đánh đổi: ~1.9× token, +~0.8 s/search cho rerank; vẫn dưới 1 cent cho cả test. Giới hạn: (1) test chỉ 6+2 câu, chênh lệch lớn có thể do mẫu nhỏ; (2) **người chấm nhãn là AI (Claude) đọc trace và toàn văn chunk được trích dẫn** — học viên cần rà soát lại `reviews.json` trước khi nộp; (3) dev được dùng để viết prompt v2/v3 (có nhắc ví dụ từ vựng như "colonies", "mercenaries" lấy từ lỗi dev) — điều này có thể làm dev lạc quan, không ảnh hưởng test vì các câu test ở chương khác; (4) model API có thể thay đổi giữa các lần chạy; (5) q11, q15 vẫn sai ở mọi cấu hình; (6) trang là trang logic Day 6, không phải trang in.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.

- Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản: không áp dụng.
- Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý: không áp dụng.
- Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên dev: không áp dụng.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | — | — | Không thực hiện |
| Ảnh, sơ đồ | — | — | Không thực hiện |
| Biểu đồ | — | — | Không thực hiện |
| Multi-hop | — | — | Không thực hiện |
| Text | — | — | Không thực hiện |
| Thiếu bằng chứng | — | — | Không thực hiện |
| **Toàn bộ** | — | — | Không thực hiện |

- Trace, nhãn và evidence từng câu; thời gian, chi phí, lỗi hệ thống: không áp dụng.
- Giới hạn, cách tái lập và đường dẫn benchmark: không áp dụng.
