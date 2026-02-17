# API Documentation

## Base URL
`http://localhost:8000/api/v1`

## Endpoints

### Chat

#### POST `/chat`
Send a message to the agent

**Request:**
```json
{
  "message": "Hello, my name is John",
  "conversation_id": "optional-id"
}

Response:

json
{
  "response": "Nice to meet you, John!",
  "conversation_id": "20240315_123456",
  "metadata": {
    "message_count": 1,
    "timestamp": "2024-03-15 12:34:56"
  }
}
Conversation
POST /conversation/end
End current conversation

Response:

json
{
  "status": "ended",
  "summary": {
    "conversation_id": "20240315_123456",
    "duration_seconds": 120.5,
    "total_messages": 10,
    "avg_response_time": 12.05
  }
}

Memory
GET /memory/{type}
Search specific memory type

Parameters:

type: episodic, semantic, or procedural

query: search query

limit: result limit (default: 5)

Response:

json
{
  "results": [
    {
      "conversation": "...",
      "conversation_summary": "...",
      "what_worked": "..."
    }
  ],
  "count": 1
}
DELETE /memory/{type}
Clear specific memory type

Response:

json
{
  "status": "cleared",
  "memory_type": "working"
}
System
GET /stats
Get agent statistics

Response:

json
{
  "initialized": true,
  "working_memory_size": 5,
  "state": {
    "session_id": "...",
    "total_messages": 10,
    "is_active": true
  }
}
GET /health
Health check

Response:

json
{
  "status": "healthy"
}
Error Responses
json
{
  "detail": "Error message description"
}
Status codes:

400: Bad Request

404: Not Found

500: Internal Server Error

text

## `docs/examples.md`
```markdown
# Usage Examples

## Basic Chat

### Python
```python
import asyncio
from agentic_memory.agent.core import MemoryAgent

async def main():
    agent = MemoryAgent()
    await agent.initialize()
    
    response = await agent.process_message("Hello!")
    print(f"AI: {response}")
    
    response = await agent.process_message("What's my name?")
    print(f"AI: {response}")
    
    await agent.end_conversation()
    await agent.shutdown()

asyncio.run(main())
CLI
bash
# Start chat
agentic-memory chat

# With verbose output
agentic-memory chat --verbose
API
bash
# Send message
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!"}'

# End conversation
curl -X POST http://localhost:8000/api/v1/conversation/end
Memory Search
Episodic Memory
python
# Find similar past conversations
result = await agent.episodic_memory.retrieve(
    "discussion about memory types",
    limit=3
)
Semantic Memory
python
# Search knowledge base
chunks = await agent.semantic_memory.retrieve(
    "cognitive architectures",
    limit=5
)
Procedural Memory
python
# Get current rules
rules = await agent.procedural_memory.retrieve()
print(rules)
Advanced Usage
Custom Configuration
python
from agentic_memory.config.settings import Settings

settings = Settings(
    MODEL_NAME="gpt-4",
    TEMPERATURE=0.5,
    MAX_CONTEXT_MEMORIES=5
)

agent = MemoryAgent(config=settings)
Multiple Conversations
python
from agentic_memory.agent.conversation import ConversationManager

agent = MemoryAgent()
await agent.initialize()

# Conversation 1
conv1 = ConversationManager(agent)
await conv1.start()
resp1 = await conv1.process("Hello")
await conv1.end()

# Conversation 2
conv2 = ConversationManager(agent)
await conv2.start()
resp2 = await conv2.process("Hi there")
await conv2.end()
Reset All Memories
bash
# CLI
agentic-memory reset

# Python
from agentic_memory.scripts.reset_memory import reset_all_memories
import asyncio

asyncio.run(reset_all_memories())
Initialize Database
bash
# CLI
agentic-memory init

# Python
from agentic_memory.scripts.init_db import init_database
import asyncio

asyncio.run(init_database())
Testing
bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=agentic_memory tests/

# Run specific test file
pytest tests/test_memory/test_episodic.py -v
Docker
bash
# Build image
docker build -t agentic-memory .

# Run with docker-compose
docker-compose up -d

# Run container
docker run -p 8000:8000 agentic-memory
Environment Variables
bash
# Copy example env
cp .env.example .env

# Edit with your settings
OPENAI_API_KEY=sk-...
WEAVIATE_HOST=localhost
MODEL_NAME=gpt-4o
Development
bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Type checking
mypy src/