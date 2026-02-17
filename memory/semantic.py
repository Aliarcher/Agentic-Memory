from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from core.interfaces.memory import SemanticMemoryInterface
from core.models.memory import SemanticChunk
from providers.weaviate import WeaviateProvider
from core.exceptions import SemanticMemoryError
from config.settings import settings

class SemanticMemory(SemanticMemoryInterface):
    """Semantic memory implementation for factual knowledge"""
    
    def __init__(self, provider: WeaviateProvider):
        self.provider = provider
        self.logger = logging.getLogger(__name__)
        self.collection_name = settings.SEMANTIC_COLLECTION
    
    async def store(self, data: Any, **kwargs) -> None:
        """Store a semantic chunk"""
        try:
            chunk = data if isinstance(data, SemanticChunk) else SemanticChunk(**data)
            
            collection = self.provider.get_collection(self.collection_name)
            
            result = collection.data.insert({
                "chunk": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
                "created_at": datetime.now().isoformat()
            })
            
            self.logger.debug(f"Stored semantic chunk from {chunk.source}")
            
        except Exception as e:
            raise SemanticMemoryError(f"Failed to store semantic chunk: {e}")
    
    async def retrieve(self, query: str, **kwargs) -> str:
        """Retrieve relevant semantic chunks"""
        try:
            limit = kwargs.get("limit", settings.SEMANTIC_CHUNK_LIMIT)
            collection = self.provider.get_collection(self.collection_name)
            
            memories = collection.query.hybrid(
                query=query,
                alpha=0.5,
                limit=limit
            )
            
            return self._format_chunks(memories.objects)
            
        except Exception as e:
            raise SemanticMemoryError(f"Failed to retrieve semantic memory: {e}")
    
    async def search(self, query: str, limit: int = 5) -> List[SemanticChunk]:
        """Search semantic memory and return structured results"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            
            results = collection.query.hybrid(
                query=query,
                alpha=0.5,
                limit=limit
            )
            
            chunks = []
            for obj in results.objects:
                props = obj.properties
                chunks.append(SemanticChunk(
                    id=str(obj.uuid),
                    content=props.get("chunk", ""),
                    source=props.get("source", ""),
                    chunk_index=props.get("chunk_index", 0),
                    metadata=props.get("metadata", {})
                ))
            
            return chunks
            
        except Exception as e:
            raise SemanticMemoryError(f"Failed to search semantic memory: {e}")
    
    async def get_by_source(self, source: str) -> List[SemanticChunk]:
        """Get all chunks from a specific source"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            
            results = collection.query.fetch_objects(
                where={
                    "path": ["source"],
                    "operator": "Equal",
                    "valueText": source
                }
            )
            
            chunks = []
            for obj in results.objects:
                props = obj.properties
                chunks.append(SemanticChunk(
                    id=str(obj.uuid),
                    content=props.get("chunk", ""),
                    source=props.get("source", ""),
                    chunk_index=props.get("chunk_index", 0),
                    metadata=props.get("metadata", {})
                ))
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.chunk_index)
            return chunks
            
        except Exception as e:
            raise SemanticMemoryError(f"Failed to get chunks by source: {e}")
    
    async def clear(self) -> None:
        """Clear all semantic memories"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            collection.data.delete_many({})
            self.logger.info("Cleared all semantic memories")
        except Exception as e:
            raise SemanticMemoryError(f"Failed to clear semantic memory: {e}")
    
    async def delete(self, chunk_id: str) -> None:
        """Delete specific chunk by ID"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            collection.data.delete_by_id(chunk_id)
            self.logger.info(f"Deleted semantic chunk: {chunk_id}")
        except Exception as e:
            raise SemanticMemoryError(f"Failed to delete chunk {chunk_id}: {e}")
    
    def _format_chunks(self, objects) -> str:
        """Format retrieved chunks into a single string"""
        chunks = []
        for i, obj in enumerate(objects):
            props = obj.properties
            content = props.get('chunk', '').strip()
            if content:
                chunks.append(f"\nCHUNK {i+1}:\n{content}")
        
        return "".join(chunks) if chunks else ""
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        try:
            collection = self.provider.get_collection(self.collection_name)
            count = collection.aggregate.over_all(total_count=True)
            
            # Get sources distribution
            sources = {}
            all_chunks = await self.search("", limit=100)  # Get recent chunks
            for chunk in all_chunks:
                sources[chunk.source] = sources.get(chunk.source, 0) + 1
            
            return {
                "total_chunks": count.total_count,
                "sources": sources,
                "collection": self.collection_name
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}