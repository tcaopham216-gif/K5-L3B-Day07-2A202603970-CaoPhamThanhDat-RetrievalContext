# Code

Đặt code của bạn tại đây hoặc mở rộng [starter](../starter/README.md). Khung agent chưa kết nối model và vector store thật.

## Bài làm Day 7 — cách chạy (Python 3.12, Windows PowerShell)

| File | Công dụng |
|---|---|
| `prepare_data.py` | Đọc corpus/benchmark Day 6 (chỉ đọc), chunk theo chiến lược B của Day 6, tạo `data/day7/*.json`, dựng lại Chroma `day6_B` |
| `common.py` | Embedder Day 6 + 4 retriever: `dense`, `bm25`, `hybrid` (RRF), `rerank` (cross-encoder) |
| `run_retrieval.py` | Benchmark retrieval/context (Hit@1, Evidence Hit@3, Context Evidence Rate, MRR@10) |
| `agent.py` | Controller gọi model thật qua OpenRouter (tool `search`/`answer`/`abstain`), dùng `starter/agentic.py::run_agent` |
| `write_reviews.py` | Nhãn chấm thủ công → `reviews.json` cho `starter/agent_metrics.py` |

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r src\requirements.txt
$env:PYTHONUTF8 = '1'
$env:OPENROUTER_API_KEY = '<key của bạn>'   # không commit
.venv\Scripts\python src\prepare_data.py --day6 "<đường dẫn repo Day 6>"
.venv\Scripts\python src\run_retrieval.py --split dev --config configs\retrieval_dev.json
.venv\Scripts\python src\agent.py --split dev --config configs\agent_dev.json
.venv\Scripts\python src\write_reviews.py
.venv\Scripts\python starter\agent_metrics.py --cases data\day7\cases.json --traces results\agent\dev\A6-rerank-k5-v3\traces.json --reviews results\agent\dev\A6-rerank-k5-v3\reviews.json --split dev --output results\agent\dev\A6-rerank-k5-v3\agent_metrics.json
# chỉ sau khi chốt cấu hình:
.venv\Scripts\python src\run_retrieval.py --split test --config configs\retrieval_test.json
.venv\Scripts\python src\agent.py --split test --config configs\agent_test.json
.venv\Scripts\python -m unittest discover -s tests -v
```

Index Chroma và cache model nằm trong `starter/data/` (không commit, tái tạo bằng `prepare_data.py`, ~75 giây). Chạy lại agent sẽ ghi đè trace và có thể cho kết quả khác một chút (model API, temperature 0).
