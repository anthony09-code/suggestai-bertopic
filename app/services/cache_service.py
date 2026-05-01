from diskcache import Cache

cache = Cache("./cache")


class CacheService:
    def get(self, key):
        return cache.get(key)

    def set(self, key, value):
        cache.set(key, value)

    def exists(self, key):
        return key in cache


cache_service = CacheService()
