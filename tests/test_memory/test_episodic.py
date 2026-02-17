import pytest
from unittest.mock import AsyncMock, Mock
from langchain_core.messages import HumanMessage, AIMessage
from memory.episodic import EpisodicMemory
from core.models.memory import EpisodicMemoryEntry

@pytest.mark.asyncio
async def test_episodic_memory_store(mock_provider, mock_llm):
    """Test storing episodic memory"""
    memory = EpisodicMemory(mock_provider, mock_llm)
    
    # Mock reflection chain
    memory.reflection_chain.ainvoke = AsyncMock(return_value={
        "context_tags": ["test", "memory"],
        "conversation_summary": "Test conversation",
        "what_worked": "Test worked",
        "what_to_avoid": "Test avoid"
    })
    
    # Mock collection
    mock_collection = Mock()
    mock_collection.data.insert = Mock(return_value="test_id")
    mock_provider.get_collection.return_value = mock_collection
    
    # Test storing
    messages = [
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there")
    ]
    
    await memory.store(messages)
    
    mock_collection.data.insert.assert_called_once()

@pytest.mark.asyncio
async def test_episodic_memory_retrieve(mock_provider, mock_llm):
    """Test retrieving episodic memory"""
    memory = EpisodicMemory(mock_provider, mock_llm)
    
    # Mock collection and query result
    mock_collection = Mock()
    mock_result = Mock()
    mock_obj = Mock()
    mock_obj.uuid = "test_uuid"
    mock_obj.properties = {
        "conversation": "Test conversation",
        "context_tags": ["test"],
        "conversation_summary": "Summary",
        "what_worked": "Worked",
        "what_to_avoid": "Avoid",
        "created_at": "2024-01-01T00:00:00",
        "access_count": 0
    }
    mock_result.objects = [mock_obj]
    mock_collection.query.hybrid = Mock(return_value=mock_result)
    mock_provider.get_collection.return_value = mock_collection
    
    result = await memory.retrieve("test query")
    
    assert result is not None
    assert result.conversation == "Test conversation"
    assert result.conversation_summary == "Summary"

@pytest.mark.asyncio
async def test_episodic_memory_search_by_tags(mock_provider, mock_llm):
    """Test searching by tags"""
    memory = EpisodicMemory(mock_provider, mock_llm)
    
    # Mock collection and query result
    mock_collection = Mock()
    mock_result = Mock()
    mock_obj = Mock()
    mock_obj.uuid = "test_uuid"
    mock_obj.properties = {
        "conversation": "Test",
        "context_tags": ["test"],
        "conversation_summary": "Summary"
    }
    mock_result.objects = [mock_obj]
    mock_collection.query.fetch_objects = Mock(return_value=mock_result)
    mock_provider.get_collection.return_value = mock_collection
    
    results = await memory.search_by_tags(["test"])
    
    assert len(results) == 1
    assert results[0].conversation == "Test"

@pytest.mark.asyncio
async def test_episodic_memory_reflect(mock_provider, mock_llm):
    """Test reflection generation"""
    memory = EpisodicMemory(mock_provider, mock_llm)
    
    expected_reflection = {
        "context_tags": ["test"],
        "conversation_summary": "Summary",
        "what_worked": "Worked",
        "what_to_avoid": "Avoid"
    }
    
    memory.reflection_chain.ainvoke = AsyncMock(return_value=expected_reflection)
    
    messages = [HumanMessage(content="Test")]
    result = await memory.reflect(messages)
    
    assert result == expected_reflection

@pytest.mark.asyncio
async def test_episodic_memory_clear(mock_provider, mock_llm):
    """Test clearing memory"""
    memory = EpisodicMemory(mock_provider, mock_llm)
    
    mock_collection = Mock()
    mock_collection.data.delete_many = Mock()
    mock_provider.get_collection.return_value = mock_collection
    
    await memory.clear()
    
    mock_collection.data.delete_many.assert_called_once()