import pytest
import asyncio
from typing import Generator, AsyncGenerator
from unittest.mock import Mock, AsyncMock

from agent.core import MemoryAgent
from config.settings import settings
from providers.weaviate import WeaviateProvider

@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = AsyncMock()
    mock.ainvoke.return_value.content = "Test response"
    return mock

@pytest.fixture
def mock_provider():
    """Mock provider for testing"""
    provider = AsyncMock(spec=WeaviateProvider)
    provider.initialize = AsyncMock()
    provider.close = AsyncMock()
    provider.health_check = AsyncMock(return_value=True)
    provider.get_collection = Mock()
    return provider

@pytest.fixture
async def agent(mock_llm, mock_provider) -> AsyncGenerator[MemoryAgent, None]:
    """Create test agent"""
    agent = MemoryAgent()
    agent.llm = mock_llm
    agent.provider = mock_provider
    agent.initialized = True
    yield agent

@pytest.fixture
def sample_conversation():
    """Sample conversation for testing"""
    return [
        ("user", "Hello, my name is John"),
        ("assistant", "Nice to meet you, John!"),
        ("user", "What's my name?"),
        ("assistant", "Your name is John.")
    ]

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()