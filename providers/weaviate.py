import weaviate
from weaviate.collections import Collection
from typing import Optional, Dict, Any
import logging
from core.interfaces.provider import MemoryProvider
from config.settings import settings

class WeaviateProvider(MemoryProvider):
    """Weaviate implementation of memory storage"""
    
    def __init__(self):
        self.client: Optional[weaviate.WeaviateClient] = None
        self.logger = logging.getLogger(__name__)
        self._collections: Dict[str, Collection] = {}
    
    async def initialize(self) -> None:
        """Initialize connection to Weaviate"""
        try:
            self.client = weaviate.connect_to_local(
                host=settings.WEAVIATE_HOST,
                port=settings.WEAVIATE_PORT,
                grpc_port=settings.WEAVIATE_GRPC_PORT
            )
            
            if not self.client.is_ready():
                raise ConnectionError("Weaviate is not ready")
            
            self.logger.info("Connected to Weaviate successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    async def close(self) -> None:
        """Close Weaviate connection"""
        if self.client:
            self.client.close()
            self.logger.info("Closed Weaviate connection")
    
    def get_collection(self, name: str) -> Collection:
        """Get or cache a collection"""
        if name not in self._collections:
            self._collections[name] = self.client.collections.get(name)
        return self._collections[name]
    
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        return self.client is not None and self.client.is_ready()