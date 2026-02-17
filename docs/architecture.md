# Agentic Memory Architecture

## Overview

The Agentic Memory system implements a cognitive architecture with four distinct memory types:

- **Working Memory**: Short-term, active context
- **Episodic Memory**: Historical experiences and learnings
- **Semantic Memory**: Factual knowledge and concepts
- **Procedural Memory**: Rules, guidelines, and behaviors

## System Architecture
┌─────────────────────────────────────────────────────┐
│ API Layer │
│ (FastAPI, CLI, Notebooks) │
└───────────────────┬─────────────────────────────────┘
│
┌───────────────────▼─────────────────────────────────┐
│ Agent Layer │
│ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│ │ Conversation │ │ MemoryAgent  │ │ State      │ │
│ │ Manager      │ │ (Core)       │ │ Management │ │
│ └──────────────┘ └──────────────┘ └────────────┘ │
└───────────────────┬─────────────────────────────────┘
│
┌───────────────────▼─────────────────────────────────┐
│ Memory Layer │
│ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│ │ Working  │ │Episodic  │ │Semantic  │ │Proced. │ │
│ │ Memory   │ │ Memory   │ │ Memory   │ │Memory  │ │
│ └──────────┘ └──────────┘ └──────────┘ └────────┘ │
└───────────────────┬─────────────────────────────────┘
│
┌───────────────────▼─────────────────────────────────┐
│ Provider Layer │
│ ┌───────────────────────────────────────────────┐ │
│ │ Weaviate Provider │ │
│ └───────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘


## Data Flow

1. **User Input** → Working Memory
2. **Parallel Retrieval**:
   - Episodic: Similar past conversations
   - Semantic: Relevant knowledge chunks
   - Procedural: Active rules
3. **Context Assembly** → LLM
4. **Response Generation** → User
5. **Post-Processing**:
   - Store in Episodic Memory
   - Update Procedural Rules
   - Clear Working Memory

## Key Components

### MemoryAgent
Orchestrates all memory systems and LLM interaction

### ConversationManager
Manages conversation lifecycle and statistics

### Memory Providers
Abstract storage implementations (Weaviate, etc.)

### Configuration
Centralized settings management with environment support