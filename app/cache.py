# app/cache.py
import hashlib
import json
import time
from functools import lru_cache, wraps
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

# In-memory cache với TTL (Time To Live)
class MemoryCache:
    def __init__(self, default_ttl: int = 3600):  # 1 hour default
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Tạo cache key từ arguments"""
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Lấy giá trị từ cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() > entry['expires_at']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Lưu giá trị vào cache"""
        ttl = ttl or self.default_ttl
        self.cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl,
            'created_at': time.time()
        }
    
    def delete(self, key: str) -> None:
        """Xóa key khỏi cache"""
        if key in self.cache:
            del self.cache[key]
    
    def clear(self) -> None:
        """Xóa toàn bộ cache"""
        self.cache.clear()
    
    def cleanup_expired(self) -> None:
        """Dọn dẹp các entry đã hết hạn"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time > entry['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Lấy thống kê cache"""
        current_time = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if current_time > entry['expires_at']
        )
        
        return {
            'total_entries': total_entries,
            'active_entries': total_entries - expired_entries,
            'expired_entries': expired_entries,
            'cache_size_mb': sum(
                len(json.dumps(entry['value'], default=str).encode())
                for entry in self.cache.values()
            ) / (1024 * 1024)
        }

# Global cache instance
memory_cache = MemoryCache(default_ttl=3600)  # 1 hour

def cached_with_ttl(ttl: int = 3600, key_prefix: str = ""):
    """Decorator để cache function với TTL"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Tạo cache key
            cache_key = f"{key_prefix}:{memory_cache._generate_key(func.__name__, *args, **kwargs)}"
            
            # Kiểm tra cache
            cached_result = memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Thực hiện function và cache kết quả
            result = await func(*args, **kwargs)
            memory_cache.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Tạo cache key
            cache_key = f"{key_prefix}:{memory_cache._generate_key(func.__name__, *args, **kwargs)}"
            
            # Kiểm tra cache
            cached_result = memory_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Thực hiện function và cache kết quả
            result = func(*args, **kwargs)
            memory_cache.set(cache_key, result, ttl)
            return result
        
        # Trả về wrapper phù hợp
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Cache cho search results
def cache_search_result(query: str, focal_node_uuid: Optional[str] = None, group_id: Optional[str] = None, ttl: int = 1800):
    """Cache kết quả search với TTL 30 phút"""
    # Tránh lỗi f-string lồng nhau bằng cách tạo chuỗi riêng để băm
    to_hash = f"{query}:{focal_node_uuid or ''}:{group_id or ''}"
    digest = hashlib.md5(to_hash.encode()).hexdigest()
    cache_key = f"search:{digest}"
    return cache_key

# Cache cho embeddings (nếu cần)
@lru_cache(maxsize=1000)
def get_embedding_cache_key(text: str) -> str:
    """Tạo cache key cho embedding"""
    return f"embedding:{hashlib.md5(text.encode()).hexdigest()}"

# Cache cho node data
def cache_node_data(node_uuid: str, ttl: int = 3600):
    """Cache dữ liệu node với TTL 1 giờ"""
    return f"node:{node_uuid}"

# Cache cho graph connections
def cache_connections(node_uuid: str, ttl: int = 1800):
    """Cache connections của node với TTL 30 phút"""
    return f"connections:{node_uuid}"

# Utility functions
def invalidate_search_cache():
    """Xóa tất cả search cache"""
    keys_to_delete = [key for key in memory_cache.cache.keys() if key.startswith('search:')]
    for key in keys_to_delete:
        memory_cache.delete(key)

def invalidate_node_cache(node_uuid: str):
    """Xóa cache của node cụ thể"""
    memory_cache.delete(cache_node_data(node_uuid))
    memory_cache.delete(cache_connections(node_uuid))

def invalidate_all_cache():
    """Xóa toàn bộ cache"""
    memory_cache.clear()

# Cache warming functions
async def warm_up_cache(graphiti, common_queries: List[str]):
    """Làm nóng cache với các query phổ biến"""
    for query in common_queries:
        try:
            await graphiti.search(query)
        except Exception as e:
            print(f"Error warming cache for query '{query}': {e}")

# Cache monitoring
def get_cache_metrics() -> Dict[str, Any]:
    """Lấy metrics của cache"""
    stats = memory_cache.get_stats()
    return {
        **stats,
        'cache_hit_rate': 'N/A',  # Cần implement counter
        'last_cleanup': datetime.now().isoformat()
    }

# Auto cleanup task
async def auto_cleanup_cache():
    """Tự động dọn dẹp cache mỗi 10 phút"""
    while True:
        await asyncio.sleep(600)  # 10 minutes
        memory_cache.cleanup_expired()
        print(f"Cache cleanup completed. Stats: {memory_cache.get_stats()}")

# Import asyncio for async functions
import asyncio
