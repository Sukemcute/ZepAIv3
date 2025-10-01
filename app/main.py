# app/main.py
import asyncio
import os
from datetime import datetime
from fastapi import FastAPI, Depends, BackgroundTasks, HTTPException
from fastapi.responses import Response
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from app.graph import get_graphiti
from app.graph import _graphiti  # for reset endpoint
from app.schemas import IngestText, IngestMessage, IngestJSON, SearchRequest
from app.cache import (
    cached_with_ttl, cache_search_result, memory_cache, 
    invalidate_search_cache, invalidate_node_cache, get_cache_metrics
)
app = FastAPI(title="Graphiti Memory Layer")

@app.post("/ingest/text")
async def ingest_text(payload: IngestText, graphiti=Depends(get_graphiti)):
    ts = datetime.fromisoformat(payload.reference_time) if payload.reference_time else datetime.utcnow()
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Ingesting text with group_id: {payload.group_id}")
    
    ep = await graphiti.add_episode(
        name=payload.name,
        episode_body=payload.text,
        source=EpisodeType.text,
        source_description=payload.source_description,
        reference_time=ts,
        group_id=payload.group_id,
    )
    
    # Invalidate search cache khi có dữ liệu mới
    invalidate_search_cache()
    
    return {
        "episode_id": ep.id if hasattr(ep, "id") else payload.name,
        "group_id": payload.group_id,
        "name": payload.name
    }

@app.post("/ingest/message")
async def ingest_message(payload: IngestMessage, graphiti=Depends(get_graphiti)):
    ts = datetime.fromisoformat(payload.reference_time) if payload.reference_time else datetime.utcnow()
    body = "\n".join(payload.messages)  # yêu cầu dạng "speaker: message" theo doc
    
    # Debug logging
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Ingesting message with group_id: {payload.group_id}, name: {payload.name}")
    
    ep = await graphiti.add_episode(
        name=payload.name,
        episode_body=body,
        source=EpisodeType.message,
        source_description=payload.source_description,
        reference_time=ts,
        group_id=payload.group_id,
    )
    
    # Invalidate search cache khi có dữ liệu mới
    invalidate_search_cache()
    
    return {
        "episode_id": ep.id if hasattr(ep, "id") else payload.name,
        "group_id": payload.group_id,
        "name": payload.name
    }

# removed duplicate old JSON ingest using payload.json

@app.post("/search")
async def search(req: SearchRequest, graphiti=Depends(get_graphiti)):
    import logging
    from openai import OpenAI
    from app.prompts import format_query_translation_prompt, PROMPT_CONFIG
    
    logger = logging.getLogger(__name__)
    
    # Kiểm tra cache trước
    cache_key = cache_search_result(req.query, req.focal_node_uuid, req.group_id)
    cached_result = memory_cache.get(cache_key)
    if cached_result is not None:
        return cached_result

    # Auto-translate non-English queries to English for better semantic search
    search_query = req.query
    try:
        # Detect if query is non-English (simple heuristic: contains non-ASCII)
        if any(ord(char) > 127 for char in req.query):
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                client = OpenAI(api_key=openai_key)
                trans_prompt = format_query_translation_prompt(req.query)
                trans_config = PROMPT_CONFIG.get("translation", {})
                
                translation = client.chat.completions.create(
                    model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                    messages=[{"role": "user", "content": trans_prompt}],
                    temperature=trans_config.get("temperature", 0.2),
                    max_tokens=trans_config.get("max_tokens", 100),
                )
                search_query = translation.choices[0].message.content.strip()
                logger.info(f"Translated query: '{req.query}' → '{search_query}'")
    except Exception as e:
        logger.warning(f"Translation failed, using original query: {e}")
        search_query = req.query

    # Hybrid search; nếu có focal_node_uuid sẽ ưu tiên kết quả gần node đó
    if req.focal_node_uuid:
        results = await graphiti.search(search_query, req.focal_node_uuid)
    else:
        results = await graphiti.search(search_query)

    # Chuẩn hoá đầu ra (ví dụ edges → fact/plaintext)
    def normalize(item):
        import logging
        logger = logging.getLogger(__name__)
        
        # item có thể là edge/node hoặc dict; ưu tiên các trường id phổ biến
        if isinstance(item, dict):
            txt = item.get("fact") or item.get("text") or item.get("name") or str(item)
            # For edges, prefer source_node_uuid over edge uuid
            ident = (
                item.get("source_node_uuid")  # Entity UUID (for EntityEdge)
                or item.get("uuid")  # Edge/Node UUID
                or item.get("node_uuid")
                or item.get("edge_id")
                or item.get("id")
            )
            grp_id = item.get("group_id") or item.get("groupId")
        else:
            txt = getattr(item, "fact", None) or getattr(item, "text", None) or getattr(item, "name", None) or str(item)
            # For EntityEdge objects, use source_node_uuid (the actual entity)
            ident = (
                getattr(item, "source_node_uuid", None)  # Entity UUID (for EntityEdge)
                or getattr(item, "uuid", None)  # Edge/Node UUID
                or getattr(item, "node_uuid", None)
                or getattr(item, "edge_id", None)
                or getattr(item, "id", None)
            )
            grp_id = getattr(item, "group_id", None) or getattr(item, "groupId", None)
        
        # Debug logging
        logger.info(f"Normalize: type={type(item).__name__}, text={txt[:50] if txt else 'N/A'}, id={ident}, group_id={grp_id}")
        
        if not grp_id:
            logger.warning(f"Search result missing group_id: {type(item)} - {txt[:50] if txt else 'N/A'}")
        if not ident:
            logger.warning(f"Search result missing ID: {type(item)} - {txt[:50] if txt else 'N/A'}, item_keys={list(item.keys()) if isinstance(item, dict) else dir(item)}")
        
        return {"text": txt, "id": ident, "group_id": grp_id}

    # Normalize then deduplicate and filter self-echoes of the query
    normalized = [normalize(r) for r in results]

    # If group_id filtering is requested but results don't have group_id, query Neo4j directly
    if req.group_id:
        missing_group_ids = [item for item in normalized if not item.get("group_id")]
        
        if missing_group_ids:
            # Fetch group_ids from Neo4j for items missing them
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Fetching group_ids from Neo4j for {len(missing_group_ids)} items")
            
            # Create a map of uuid -> group_id from Neo4j
            uuids = [item["id"] for item in missing_group_ids if item.get("id")]
            if uuids:
                query = """
                MATCH (n)
                WHERE n.uuid IN $uuids
                RETURN n.uuid as uuid, n.group_id as group_id
                """
                group_id_map = {}
                async with graphiti.driver.session() as session:
                    result = await session.run(query, {"uuids": uuids})
                    async for record in result:
                        if record["group_id"]:
                            group_id_map[record["uuid"]] = record["group_id"]
                
                # Update normalized items with fetched group_ids
                for item in normalized:
                    if not item.get("group_id") and item.get("id") in group_id_map:
                        item["group_id"] = group_id_map[item["id"]]
        
        # Now filter by group_id
        normalized = [item for item in normalized if item.get("group_id") == req.group_id]

    seen = set()
    deduped = []
    for item in normalized:
        key = item.get("id") or item.get("text")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    # Filter out items that are just the query echoed back
    q = (req.query or "").strip()
    q_variants = {q, f"user: {q}", f"assistant: {q}"}
    filtered = [it for it in deduped if (it.get("text") or "").strip() not in q_variants]

    # Cache kết quả với TTL 30 phút
    result = {"results": filtered}
    memory_cache.set(cache_key, result, ttl=1800)
    
    return result
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "graphiti-memory-layer",
        "endpoints": {
            "/ingest/text": "POST - Ingest plain text",
            "/ingest/message": "POST - Ingest conversation messages",
            "/search": "POST - Search knowledge graph",
            "/export/{group_id}": "GET - Export conversation to JSON"
        }
    }

@app.get("/export/{group_id}")
async def export_conversation(group_id: str, graphiti=Depends(get_graphiti)):
    """Export conversation to JSON for backup/sharing"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Query all entities for this group
        query_entities = """
        MATCH (e:Entity)
        WHERE e.group_id = $group_id
        RETURN e.uuid AS uuid,
               e.name AS name,
               e.summary AS summary,
               e.created_at AS created_at
        ORDER BY e.created_at ASC
        """
        
        entities = []
        
        async with graphiti.driver.session() as session:
            result = await session.run(query_entities, {"group_id": group_id})
            async for record in result:
                entities.append({
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "summary": record["summary"],
                    "created_at": str(record["created_at"]) if record["created_at"] else None,
                })
        
        export_data = {
            "group_id": group_id,
            "exported_at": datetime.utcnow().isoformat(),
            "entity_count": len(entities),
            "entities": entities,
            "version": "1.0"
        }
        
        logger.info(f"Exported {group_id}: {len(entities)} entities")
        
        # Return with proper UTF-8 encoding (no Unicode escaping)
        import json
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        
        # Encode to UTF-8 bytes to preserve Vietnamese characters
        return Response(
            content=json_str.encode('utf-8'),
            media_type="application/json; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/config/neo4j")
async def get_neo4j_config():
    return {
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USER": os.getenv("NEO4J_USER"),
        "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
    }

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.post("/ingest/json")
async def ingest_json(payload: IngestJSON, graphiti=Depends(get_graphiti)):
    ts = datetime.fromisoformat(payload.reference_time) if payload.reference_time else datetime.utcnow()
    ep = await graphiti.add_episode(
        name=payload.name,
        episode_body=payload.data,   # <<< đổi từ payload.json -> payload.data
        source=EpisodeType.json,
        source_description=payload.source_description,
        reference_time=ts,
        group_id=payload.group_id,
    )
    
    # Invalidate search cache khi có dữ liệu mới
    invalidate_search_cache()
    
    return {"episode_id": getattr(ep, "id", payload.name)}

# Cache management endpoints
@app.get("/cache/stats")
async def get_cache_stats():
    """Lấy thống kê cache"""
    return get_cache_metrics()

@app.post("/cache/clear")
async def clear_cache():
    """Xóa toàn bộ cache"""
    from app.cache import invalidate_all_cache
    invalidate_all_cache()
    return {"message": "Cache cleared successfully"}

@app.post("/cache/clear-search")
async def clear_search_cache():
    """Xóa chỉ search cache"""
    invalidate_search_cache()
    return {"message": "Search cache cleared successfully"}

@app.post("/cache/clear-node/{node_uuid}")
async def clear_node_cache(node_uuid: str):
    """Xóa cache của node cụ thể"""
    invalidate_node_cache(node_uuid)
    return {"message": f"Cache for node {node_uuid} cleared successfully"}

@app.get("/cache/health")
async def cache_health():
    """Kiểm tra sức khỏe cache"""
    stats = get_cache_metrics()
    return {
        "status": "healthy" if stats["active_entries"] > 0 else "empty",
        "stats": stats
    }

@app.post("/config/reload-neo4j")
async def reload_neo4j_config():
    # Đặt lại singleton để lần gọi tiếp theo tạo kết nối mới theo .env
    from app.graph import reset_graphiti_cache
    reset_graphiti_cache()
    return {"message": "Neo4j config reloaded. Restart next request will create a new connection."}

@app.get("/debug/episodes/{group_id}")
async def debug_episodes_by_group(group_id: str, graphiti=Depends(get_graphiti)):
    """Debug endpoint to check entities in Neo4j by group_id (group_id is stored in Entity nodes, not EpisodeNode)"""
    try:
        # Query Neo4j directly for Entity nodes with this group_id
        query = """
        MATCH (e:Entity)
        WHERE e.group_id = $group_id
        RETURN e.uuid as uuid, e.name as name, e.group_id as group_id, 
               e.summary as summary, e.created_at as created_at
        ORDER BY e.created_at DESC
        LIMIT 50
        """
        
        records = []
        async with graphiti.driver.session() as session:
            result = await session.run(query, {"group_id": group_id})
            async for record in result:
                records.append({
                    "uuid": record["uuid"],
                    "name": record["name"],
                    "group_id": record["group_id"],
                    "summary": record["summary"][:200] if record["summary"] else None,
                    "created_at": str(record["created_at"]).split("T")[0]
                })
        # Query Neo4j for relationships between entities with this group_id
        query = """
        MATCH (e1:Entity)-[r]-(e2:Entity)
        WHERE e1.group_id = $group_id
        RETURN e1.name as source, type(r) as rel_type, 
               properties(r) as rel_props, e2.name as target,
               e1.uuid as source_uuid, e2.uuid as target_uuid
        LIMIT 50
        """
        
        records = []
        async with graphiti.driver.session() as session:
            result = await session.run(query, {"group_id": group_id})
            async for record in result:
                records.append({
                    "source": record["source"],
                    "relationship": record["rel_type"],
                    "properties": record["rel_props"],
                    "target": record["target"],
                    "source_uuid": record["source_uuid"],
                    "target_uuid": record["target_uuid"]
                })
        
        return {
            "group_id": group_id,
            "count": len(records),
            "relationships": records
        }
    except Exception as e:
        return {"error": str(e), "group_id": group_id}