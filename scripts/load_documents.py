import asyncio
import logging
from pathlib import Path
from chunking_evaluation.chunking import RecursiveTokenChunker
from langchain_community.document_loaders import PyPDFLoader

from providers.weaviate import WeaviateProvider
from config.settings import settings

logger = logging.getLogger(__name__)

async def load_documents(pdf_path: Path):
    """Load and chunk documents into semantic memory"""
    
    if not pdf_path.exists():
        logger.error(f"PDF file not found: {pdf_path}")
        return
    
    provider = WeaviateProvider()
    
    try:
        await provider.initialize()
        
        # Load PDF
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        pages = []
        for page in loader.load():
            pages.append(page)
        
        # Combine all pages
        document = " ".join(page.page_content for page in pages)
        
        # Chunk document
        chunker = RecursiveTokenChunker(
            chunk_size=800,
            chunk_overlap=0,
            length_function=len,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""]
        )
        
        chunks = chunker.split_text(document)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Store in database
        collection = provider.get_collection(settings.SEMANTIC_COLLECTION)
        
        for i, chunk in enumerate(chunks):
            collection.data.insert({
                "chunk": chunk,
                "source": pdf_path.name,
                "chunk_index": i
            })
            
            if (i + 1) % 10 == 0:
                logger.info(f"Inserted {i + 1}/{len(chunks)} chunks")
        
        logger.info(f"Successfully loaded {len(chunks)} chunks into semantic memory")
        
        await provider.close()
        
    except Exception as e:
        logger.error(f"Failed to load documents: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pdf_path = settings.DOCUMENTS_DIR / "CoALA_Paper.pdf"
    asyncio.run(load_documents(pdf_path))