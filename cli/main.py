import asyncio
import click
import logging
from pathlib import Path

from agent.core import MemoryAgent
from config.settings import settings
from scripts.init_db import init_database
from scripts.load_documents import load_documents

@click.group()
def cli():
    """Agentic Memory CLI"""
    pass

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def chat(verbose):
    """Start an interactive chat session"""
    if verbose:
        logging.basicConfig(level=logging.INFO)
    
    async def run_chat():
        agent = MemoryAgent()
        await agent.initialize()
        
        click.echo("🤖 Agentic Memory Chat Started (type 'exit' to quit)")
        click.echo("-" * 50)
        
        try:
            while True:
                user_input = click.prompt("\nYou", prompt_suffix="> ")
                
                if user_input.lower() in ['exit', 'quit']:
                    await agent.end_conversation()
                    await agent.shutdown()
                    click.echo("Goodbye! 👋")
                    break
                
                response = await agent.process_message(user_input)
                click.echo(f"\nAI: {response}")
                
        except KeyboardInterrupt:
            click.echo("\n\nInterrupted. Shutting down...")
            await agent.end_conversation()
            await agent.shutdown()
    
    asyncio.run(run_chat())

@cli.command()
def init():
    """Initialize database and load documents"""
    async def run_init():
        click.echo("Initializing database...")
        await init_database()
        
        click.echo("Loading documents...")
        await load_documents(settings.DOCUMENTS_DIR / "CoALA_Paper.pdf")
        
        click.echo("✅ Initialization complete!")
    
    asyncio.run(run_init())

@cli.command()
def reset():
    """Reset all memories"""
    from scripts.reset_memory import reset_all_memories
    asyncio.run(reset_all_memories())
    click.echo("✅ All memories reset")

if __name__ == '__main__':
    cli()