class AgenticMemoryError(Exception):
    """Base exception for agentic memory system"""
    pass

class MemoryError(AgenticMemoryError):
    """Base memory-related error"""
    pass

class WorkingMemoryError(MemoryError):
    """Working memory specific error"""
    pass

class EpisodicMemoryError(MemoryError):
    """Episodic memory specific error"""
    pass

class SemanticMemoryError(MemoryError):
    """Semantic memory specific error"""
    pass

class ProceduralMemoryError(MemoryError):
    """Procedural memory specific error"""
    pass

class ProviderError(AgenticMemoryError):
    """Provider-related error"""
    pass

class WeaviateError(ProviderError):
    """Weaviate specific error"""
    pass

class ConfigurationError(AgenticMemoryError):
    """Configuration error"""
    pass

class AgentError(AgenticMemoryError):
    """Agent operation error"""
    pass

class RetrievalError(AgenticMemoryError):
    """Memory retrieval error"""
    pass

class StorageError(AgenticMemoryError):
    """Memory storage error"""
    pass