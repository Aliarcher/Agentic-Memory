import asyncio
import logging
from weaviate.classes.config import Property, DataType, Configure

from providers.weaviate import WeaviateProvider
from config.settings import settings

logger = logging.getLogger(__name__)

async def init_database():
    """Initialize database collections"""
    provider = WeaviateProvider()
    
    try:
        await provider.initialize()
        
        # Create episodic memory collection
        if provider.client.collections.exists(settings.EPISODIC_COLLECTION):
            provider.client.collections.delete(settings.EPISODIC_COLLECTION)
            logger.info(f"Deleted existing collection: {settings.EPISODIC_COLLECTION}")
        
        episodic = provider.client.collections.create(
            name=settings.EPISODIC_COLLECTION,
            description="Collection containing historical chat interactions and takeaways",
            vectorizer_config=[
                Configure.NamedVectors.text2vec_ollama(
                    name="title_vector",
                    source_properties=["title"],
                    api_endpoint="http://host.docker.internal:11434",
                    model="nomic-embed-text",
                )
            ],
            properties=[
                Property(name="conversation", data_type=DataType.TEXT),
                Property(name="context_tags", data_type=DataType.TEXT_ARRAY),
                Property(name="conversation_summary", data_type=DataType.TEXT),
                Property(name="what_worked", data_type=DataType.TEXT),
                Property(name="what_to_avoid", data_type=DataType.TEXT),
            ]
        )
        logger.info(f"Created episodic memory collection: {episodic.name}")
        
        # Create semantic memory collection
        if provider.client.collections.exists(settings.SEMANTIC_COLLECTION):
            provider.client.collections.delete(settings.SEMANTIC_COLLECTION)
            logger.info(f"Deleted existing collection: {settings.SEMANTIC_COLLECTION}")
        
        semantic = provider.client.collections.create(
            name=settings.SEMANTIC_COLLECTION,
            description="Collection containing paper chunks",
            vectorizer_config=[
                Configure.NamedVectors.text2vec_ollama(
                    name="title_vector",
                    source_properties=["title"],
                    api_endpoint="http://host.docker.internal:11434",
                    model="nomic-embed-text",
                )
            ],
            properties=[
                Property(name="chunk", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
            ]
        )
        logger.info(f"Created semantic memory collection: {semantic.name}")
        
        await provider.close()
        logger.info("Database initialization complete")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(init_database())