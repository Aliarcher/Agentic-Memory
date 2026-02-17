from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PROCEDURAL_MEMORY_PATH: Path = DATA_DIR / "procedural" / "procedural_memory.txt"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    
    # Model settings
    OPENAI_API_KEY: Optional[str] = None
    MODEL_NAME: str = "gpt-4o"
    TEMPERATURE: float = 0.7
    
    # Weaviate settings
    WEAVIATE_HOST: str = "localhost"
    WEAVIATE_PORT: int = 8080
    WEAVIATE_GRPC_PORT: int = 50051
    EPISODIC_COLLECTION: str = "episodic_memory"
    SEMANTIC_COLLECTION: str = "CoALA_Paper"
    
    # Memory settings
    MAX_CONTEXT_MEMORIES: int = 3
    SEMANTIC_CHUNK_LIMIT: int = 15
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()