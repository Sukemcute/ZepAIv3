# app/graph.py
import os
from pathlib import Path
from dotenv import load_dotenv
from graphiti_core import Graphiti
from app.cache import cached_with_ttl

# Graphiti sẽ dùng mặc định OpenAI cho LLM/embeddings nếu có OPENAI_API_KEY
# Bạn có thể truyền client tuỳ chỉnh theo LLM Configuration doc khi cần.

# Load .env from project (memory_layer/.env)
# override=True để .env ghi đè các biến môi trường cũ (vd còn sót cấu hình Aura)
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)

_graphiti: Graphiti | None = None

async def get_graphiti() -> Graphiti:
    """Lấy Graphiti instance với caching"""
    global _graphiti
    if _graphiti is None:
        # Graphiti dùng OpenAI embeddings mặc định.
        # Ta cho phép cấu hình model qua env để đảm bảo dùng model mới.
        # Nếu biến không được set, đặt mặc định an toàn.
        os.environ.setdefault("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))

        _graphiti = Graphiti(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "neo4j"),
        )
    return _graphiti

@cached_with_ttl(ttl=3600, key_prefix="graphiti_search")
async def cached_search(query: str, focal_node_uuid: str = None):
    """Cached search function"""
    graphiti = await get_graphiti()
    if focal_node_uuid:
        return await graphiti.search(query, focal_node_uuid)
    else:
        return await graphiti.search(query)

@cached_with_ttl(ttl=1800, key_prefix="graphiti_node")
async def cached_get_node(node_uuid: str):
    """Cached get node function"""
    graphiti = await get_graphiti()
    # Giả sử có method get_node, nếu không có thì bỏ qua
    try:
        return await graphiti.get_node(node_uuid)
    except AttributeError:
        # Nếu method không tồn tại, trả về None
        return None

def reset_graphiti_cache():
    """Reset Graphiti instance cache"""
    global _graphiti
    _graphiti = None
