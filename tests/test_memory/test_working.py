import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from memory.working import WorkingMemory

@pytest.mark.asyncio
async def test_working_memory_store_and_retrieve():
    """Test storing and retrieving messages"""
    memory = WorkingMemory(max_size=10)
    
    # Store messages
    await memory.store(HumanMessage(content="Hello"))
    await memory.store(AIMessage(content="Hi there"))
    
    # Retrieve all
    messages = await memory.retrieve()
    assert len(messages) == 2
    assert messages[0].content == "Hello"
    assert messages[1].content == "Hi there"

@pytest.mark.asyncio
async def test_working_memory_max_size():
    """Test max size limiting"""
    memory = WorkingMemory(max_size=3)
    
    # Store 5 messages
    for i in range(5):
        await memory.store(HumanMessage(content=f"Message {i}"))
    
    messages = await memory.retrieve()
    assert len(messages) == 3
    assert messages[0].content == "Message 2"
    assert messages[-1].content == "Message 4"

@pytest.mark.asyncio
async def test_working_memory_filter_by_type():
    """Test filtering by message type"""
    memory = WorkingMemory()
    
    await memory.store(SystemMessage(content="System"))
    await memory.store(HumanMessage(content="Human"))
    await memory.store(AIMessage(content="AI"))
    
    # Filter by type
    system_msgs = await memory.retrieve(type="system")
    assert len(system_msgs) == 1
    assert system_msgs[0].content == "System"
    
    user_msgs = await memory.retrieve(type="user")
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "Human"

@pytest.mark.asyncio
async def test_working_memory_clear():
    """Test clearing memory"""
    memory = WorkingMemory()
    
    await memory.store(HumanMessage(content="Test"))
    assert len(await memory.retrieve()) == 1
    
    await memory.clear()
    assert len(await memory.retrieve()) == 0

@pytest.mark.asyncio
async def test_working_memory_remove_last():
    """Test removing last messages"""
    memory = WorkingMemory()
    
    for i in range(5):
        await memory.store(HumanMessage(content=f"Msg {i}"))
    
    await memory.remove_last(2)
    messages = await memory.retrieve()
    assert len(messages) == 3
    assert messages[-1].content == "Msg 2"

@pytest.mark.asyncio
async def test_working_memory_search():
    """Test searching messages"""
    memory = WorkingMemory()
    
    await memory.store(HumanMessage(content="Hello world"))
    await memory.store(AIMessage(content="Hi there"))
    await memory.store(HumanMessage(content="Another hello"))
    
    results = await memory.search("hello")
    assert len(results) == 2
    assert all("hello" in msg.content.lower() for msg in results)

@pytest.mark.asyncio
async def test_working_memory_get_last():
    """Test getting last messages"""
    memory = WorkingMemory()
    
    await memory.store_user("User message")
    await memory.store_ai("AI response")
    
    last_user = await memory.get_last_user_message()
    assert last_user.content == "User message"
    
    last_ai = await memory.get_last_ai_message()
    assert last_ai.content == "AI response"

@pytest.mark.asyncio
async def test_working_memory_metadata():
    """Test metadata tracking"""
    memory = WorkingMemory()
    
    await memory.store_system("System")
    await memory.store_user("User 1")
    await memory.store_ai("AI 1")
    await memory.store_user("User 2")
    
    metadata = memory.get_metadata()
    assert metadata["total_messages"] == 4
    assert metadata["system_prompts"] == 1
    assert metadata["user_messages"] == 2
    assert metadata["ai_messages"] == 1