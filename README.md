# 🧠 ZepAI - AI Memory Layer

ZepAI là một hệ thống AI Memory Layer sử dụng Knowledge Graph để lưu trữ và truy xuất thông tin từ các cuộc hội thoại.

## ✨ Tính Năng

- **🗄️ Knowledge Graph Storage**: Lưu trữ thông tin với Neo4j + Graphiti
- **🔍 Semantic Search**: Tìm kiếm thông minh trong knowledge graph
- **💬 Chat với Memory**: AI assistant nhớ được cuộc trò chuyện
- **📊 Multi-tier Memory**: Short-term, mid-term, long-term memory
- **🌐 Multilingual**: Hỗ trợ tiếng Việt và các ngôn ngữ khác
- **⚡ Caching**: Tối ưu hiệu suất với in-memory cache

## 🚀 Cài Đặt

### 1. Clone Repository

```bash
git clone https://github.com/Sukemcute/ZepAIv3.git
cd ZepAIv3
```

### 2. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

### 3. Cấu Hình Environment

```bash
# Copy file cấu hình mẫu
cp .env.example .env

# Chỉnh sửa .env với thông tin của bạn
```

**Cần cấu hình:**
- `OPENAI_API_KEY`: API key từ OpenAI
- `NEO4J_URI`: URI của Neo4j database (có thể dùng Neo4j Aura miễn phí)
- `NEO4J_PASSWORD`: Mật khẩu Neo4j

### 4. Chạy Ứng Dụng

```bash
# Chạy API server
uvicorn app.main:app --reload --port 8000

# Chạy Streamlit UI (terminal khác)
streamlit run ui/streamlit_app.py
```

## 📖 Sử Dụng

### API Endpoints

- `POST /ingest/text` - Nạp text vào knowledge graph
- `POST /ingest/message` - Nạp conversation vào knowledge graph  
- `POST /search` - Tìm kiếm trong knowledge graph
- `GET /export/{group_id}` - Export conversation to JSON

### Streamlit UI

Truy cập `http://localhost:8501` để sử dụng giao diện chat với memory.

## 🏗️ Kiến Trúc

```
ZepAIv3/
├── app/
│   ├── main.py           # FastAPI server
│   ├── graph.py          # Neo4j/Graphiti management
│   ├── cache.py          # Caching system
│   ├── schemas.py        # Pydantic models
│   ├── prompts.py        # Prompt engineering
│   └── importance.py     # Fact importance scoring
├── ui/
│   └── streamlit_app.py  # Streamlit interface
├── query_graph.py        # Debug tool
└── requirements.txt      # Dependencies
```

## 🔧 Configuration

Xem file `.env.example` để biết các biến môi trường cần thiết.

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Repo gốc**: Dựa trên [NguyenTrinh3008/ZepAI](https://github.com/NguyenTrinh3008/ZepAI)