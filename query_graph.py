"""
Query and visualize graph for a specific group_id
Usage: python query_graph.py chat-d1344a24-dcd6-45fd-94ee-b8792982f0be
"""

import asyncio
import sys
from app.graph import get_graphiti
import json

async def query_graph(group_id: str):
    """Query all nodes and relationships for a group_id"""
    graphiti = await get_graphiti()
    
    queries = {
        "entities": """
        MATCH (e:Entity)
        WHERE e.group_id = $group_id
        RETURN e.name AS name, 
               e.summary AS summary,
               e.created_at AS created,
               e.uuid AS uuid
        ORDER BY e.created_at DESC
        """,
        
        "relationships": """
        MATCH (e1:Entity)-[r]->(e2:Entity)
        WHERE e1.group_id = $group_id
          AND e2.group_id = $group_id
        RETURN e1.name AS from,
               type(r) AS type,
               r.fact AS fact,
               e2.name AS to
        LIMIT 100
        """,
        
        "statistics": """
        MATCH (n)
        WHERE n.group_id = $group_id
        WITH labels(n) AS labels, count(*) AS count
        RETURN labels, count
        ORDER BY count DESC
        """
    }
    
    results = {}
    
    async with graphiti.driver.session() as session:
        for query_name, query in queries.items():
            print(f"\n{'='*60}")
            print(f"Query: {query_name.upper()}")
            print('='*60)
            
            result = await session.run(query, {"group_id": group_id})
            records = []
            
            async for record in result:
                record_dict = dict(record)
                records.append(record_dict)
                print(json.dumps(record_dict, indent=2, default=str))
            
            results[query_name] = records
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Group ID: {group_id}")
    print(f"Total Entities: {len(results['entities'])}")
    print(f"Total Relationships: {len(results['relationships'])}")
    print("\nNode Types:")
    for stat in results['statistics']:
        print(f"  {stat['labels']}: {stat['count']}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_graph.py <group_id>")
        print("Example: python query_graph.py chat-d1344a24-dcd6-45fd-94ee-b8792982f0be")
        sys.exit(1)
    
    group_id = sys.argv[1]
    asyncio.run(query_graph(group_id))
