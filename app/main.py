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

@app.post("/debug/embed")
async def debug_embed(payload: dict):
    """Compute embeddings and cosine similarities for given texts.
    Payload examples:
      {"texts": ["xin chào", "chào bạn", "tôi ăn cơm"]}
      or {"pairs": [["a", "b"], ["a", "c"]]}
    Returns normalized vectors and cosine similarities.
    """
    try:
        from openai import OpenAI
        import math
        import os

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL" ,"text-embedding-3-small"))

        def l2_normalize(vec: list[float]) -> list[float]:
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            return [v / norm for v in vec]

        def cosine(a: list[float], b: list[float]) -> float:
            return sum(x*y for x, y in zip(a, b))

        texts = payload.get("texts")
        pairs = payload.get("pairs")

        results = {}

        if texts and isinstance(texts, list):
            emb = client.embeddings.create(model=model, input=texts)
            vecs = [l2_normalize(e.embedding) for e in emb.data]
            results["texts"] = texts
            results["vectors"] = vecs
            # pairwise cosine
            sims = []
            for i in range(len(vecs)):
                row = []
                for j in range(len(vecs)):
                    row.append(round(cosine(vecs[i], vecs[j]), 6))
                sims.append(row)
            results["cosine_matrix"] = sims

        if pairs and isinstance(pairs, list):
            flat = []
            idx = []
            for a, b in pairs:
                idx.append((len(flat), len(flat)+1))
                flat.extend([a, b])
            emb = client.embeddings.create(model=model, input=flat)
            vecs = [l2_normalize(e.embedding) for e in emb.data]
            pair_sims = []
            for i_a, i_b in idx:
                pair_sims.append(round(cosine(vecs[i_a], vecs[i_b]), 6))
            results["pairs"] = pairs
            results["pair_similarities"] = pair_sims

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding debug failed: {str(e)}")

@app.post("/embed/json")
async def embed_json(payload: dict):
    """Add embeddings into provided JSON.
    Supported inputs:
      - {"format": "mid_term", "mid_term": [{"id":..., "text":..., "metadata": {...}}, ...]}
      - {"texts": ["..."]}
      - {"entities": [{"uuid":..., "summary":...}, ...]}  # raw format
    Returns the same structure with an added field "embedding" (L2-normalized floats).
    """
    try:
        from openai import OpenAI
        import os, math

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL" ,"text-embedding-3-small"))

        def l2_normalize(vec: list[float]) -> list[float]:
            norm = math.sqrt(sum(v*v for v in vec)) or 1.0
            return [v / norm for v in vec]

        # Collect texts depending on structure
        to_embed: list[str] = []
        index_map: list[tuple[str, int]] = []  # (section, idx)

        fmt = payload.get("format")
        if fmt == "mid_term" and isinstance(payload.get("mid_term"), list):
            for i, entry in enumerate(payload["mid_term"]):
                text = (entry or {}).get("text")
                if isinstance(text, str) and text.strip():
                    index_map.append(("mid_term", i))
                    to_embed.append(text)
        elif isinstance(payload.get("texts"), list):
            for i, text in enumerate(payload["texts"]):
                if isinstance(text, str) and text.strip():
                    index_map.append(("texts", i))
                    to_embed.append(text)
        elif isinstance(payload.get("entities"), list):
            for i, ent in enumerate(payload["entities"]):
                text = (ent or {}).get("summary") or (ent or {}).get("name")
                if isinstance(text, str) and text.strip():
                    index_map.append(("entities", i))
                    to_embed.append(text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported JSON shape. Use format=mid_term, or texts, or entities.")

        # Batch embed (OpenAI supports batching; chunk to avoid oversized requests)
        vectors: list[list[float]] = []
        batch_size = 512
        for i in range(0, len(to_embed), batch_size):
            chunk = to_embed[i:i+batch_size]
            emb = client.embeddings.create(model=model, input=chunk)
            vectors.extend([l2_normalize(e.embedding) for e in emb.data])

        # Write back embeddings
        for (section, idx), vec in zip(index_map, vectors):
            if section == "mid_term":
                payload["mid_term"][idx]["embedding"] = vec
            elif section == "texts":
                # Create parallel array if not exists
                if "embeddings" not in payload:
                    payload["embeddings"] = [None] * len(payload["texts"])
                payload["embeddings"][idx] = vec
            elif section == "entities":
                payload["entities"][idx]["embedding"] = vec

        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed JSON failed: {str(e)}")
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

def _parse_iso_ts(value: str | None) -> datetime:
    """Robust ISO8601 parser that accepts 'Z' suffix and various formats.
    Falls back to UTC now on failure.
    """
    if not value:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        try:
            # Try common alternate formats
            from dateutil import parser as dtparser  # optional dependency
            return dtparser.parse(value)
        except Exception:
            return datetime.utcnow()


@app.post("/import/conversation")
async def import_conversation(payload: dict, graphiti=Depends(get_graphiti)):
    """
    Import conversation from JSON file (mid_term or raw format)
    
    Supports:
    - mid_term format: {format: "mid_term", mid_term: [...]}
    - raw format: {format: "raw", entities: [...]}
    """
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        file_format = payload.get("format", "unknown")
        imported_count = 0
        failed_count = 0
        item_errors: list[dict] = []
        
        if file_format == "mid_term":
            # Import mid_term format
            entries = payload.get("mid_term", [])
            group_id = payload.get("group_id", "imported")
            
            logger.info(f"Importing {len(entries)} mid_term entries to group_id: {group_id}")
            
            for entry in entries:
                try:
                    text = entry.get("text", "")
                    metadata = entry.get("metadata", {})
                    timestamp = metadata.get("timestamp", None)
                    
                    if not text or len(text.strip()) < 3:
                        continue
                    
                    # Ingest each entry as text episode
                    ts = _parse_iso_ts(timestamp)
                    
                    await graphiti.add_episode(
                        name=f"imported_{entry.get('id', 'unknown')}",
                        episode_body=text,
                        source=EpisodeType.text,
                        source_description=f"imported_from_{metadata.get('source', 'unknown')}",
                        reference_time=ts,
                        group_id=group_id,
                    )
                    
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to import entry: {e}")
                    item_errors.append({"id": entry.get("id"), "error": str(e)})
                    failed_count += 1
                    continue
            
            invalidate_search_cache()
            
            return {
                "success": True,
                "format": "mid_term",
                "imported": imported_count,
                "failed": failed_count,
                "group_id": group_id,
                "errors": item_errors[:10]  # return first 10 errors for brevity
            }
            
        elif file_format == "raw":
            # Import raw format
            entities = payload.get("entities", [])
            group_id = payload.get("group_id", "imported")
            
            logger.info(f"Importing {len(entities)} raw entities to group_id: {group_id}")
            
            for entity in entities:
                try:
                    summary = entity.get("summary", "") or entity.get("name", "")
                    created_at = entity.get("created_at", None)
                    
                    if not summary or len(summary.strip()) < 3:
                        continue
                    
                    ts = _parse_iso_ts(created_at)
                    
                    await graphiti.add_episode(
                        name=f"imported_{entity.get('uuid', 'unknown')}",
                        episode_body=summary,
                        source=EpisodeType.text,
                        source_description="imported_from_raw",
                        reference_time=ts,
                        group_id=group_id,
                    )
                    
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to import entity: {e}")
                    item_errors.append({"uuid": entity.get("uuid"), "error": str(e)})
                    failed_count += 1
                    continue
            
            invalidate_search_cache()
            
            return {
                "success": True,
                "format": "raw",
                "imported": imported_count,
                "failed": failed_count,
                "group_id": group_id,
                "errors": item_errors[:10]
            }
        
        else:
            return {
                "success": False,
                "error": f"Unsupported format: {file_format}. Supported: mid_term, raw",
            }
            
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return {"success": False, "error": str(e)}

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