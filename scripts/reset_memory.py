import asyncio
import logging
from pathlib import Path

from providers.weaviate import WeaviateProvider
from config.settings import settings

logger = logging.getLogger(__name__)

async def reset_all_memories():
    """Reset all memory systems"""
    
    # Reset Weaviate collections
    provider = WeaviateProvider()
    
    try:
        await provider.initialize()
        
        # Delete episodic memory
        if provider.client.collections.exists(settings.EPISODIC_COLLECTION):
            provider.client.collections.delete(settings.EPISODIC_COLLECTION)
            logger.info(f"Deleted episodic memory collection")
        
        # Delete semantic memory
        if provider.client.collections.exists(settings.SEMANTIC_COLLECTION):
            provider.client.collections.delete(settings.SEMANTIC_COLLECTION)
            logger.info(f"Deleted semantic memory collection")
        
        await provider.close()
        
    except Exception as e:
        logger.error(f"Failed to reset Weaviate: {e}")
    
    # Reset procedural memory file
    procedural_path = settings.PROCEDURAL_MEMORY_PATH
    if procedural_path.exists():
        # Create backup
        backup_path = procedural_path.with_suffix('.txt.bak')
        procedural_path.rename(backup_path)
        logger.info(f"Backed up procedural memory to {backup_path}")
    
    # Create new procedural memory file with defaults
    default_rules = [
        "1. Maintain conversation context by recalling previous interactions - Builds rapport and shows attention to user preferences over time.",
        "2. Use clear and concise language to convey information - Enhances understanding and avoids confusion.",
        "3. Offer structured breakdowns for complex topics - Facilitates comprehension and highlights key roles and functions.",
        "4. Ask clarifying questions when user requests are ambiguous - Ensures accurate assistance and reduces misunderstandings.",
        "5. Provide step-by-step guidance for complex tasks - Facilitates user comprehension and successful task completion."
    ]
    
    procedural_path.parent.mkdir(parents=True, exist_ok=True)
    with open(procedural_path, "w") as f:
        f.write("\n".join(default_rules))
    
    logger.info(f"Created new procedural memory file with {len(default_rules)} default rules")
    logger.info("All memories reset successfully")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(reset_all_memories())