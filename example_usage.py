"""
Example usage script for the RAG system.
Demonstrates basic functionality without UI.
"""
from rag_pipeline import RAGPipeline
from query_processor import QueryProcessor, DocumentSummarizer, LLMManager
from config import Config


def main():
    """Example usage of the RAG system."""
    
    # Validate configuration
    Config.validate()
    
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Example 1: Load documents from file
    print("\n=== Example 1: Loading Documents ===")
    try:
        # Replace with your document path
        document_path = "example_document.pdf"  # or "example_document.txt"
        
        print(f"Loading document: {document_path}")
        pipeline.ingest_documents(
            source=document_path,
            source_type="file"
        )
        print("Documents loaded successfully!")
    except FileNotFoundError:
        print(f"Document not found: {document_path}")
        print("Please provide a valid document path or skip this example.")
        return
    except Exception as e:
        print(f"Error loading documents: {e}")
        return
    
    # Example 2: Question Answering
    print("\n=== Example 2: Question Answering ===")
    query_processor = QueryProcessor(pipeline)
    
    questions = [
        "What is the main topic of this document?",
        "What are the key points discussed?",
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        try:
            result = query_processor.process_query(
                question=question,
                include_walkthrough=True,
                k=4
            )
            
            print(f"Answer: {result['answer']}")
            print(f"Sources: {result['num_sources']} documents retrieved")
            
            if 'walkthrough' in result:
                print(f"\nWalkthrough:\n{result['walkthrough'][:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Example 3: Document Summarization
    print("\n=== Example 3: Document Summarization ===")
    try:
        summarizer = DocumentSummarizer(pipeline, LLMManager())
        
        print("Generating summary...")
        summary_result = summarizer.summarize_documents(
            summary_type="map_reduce"
        )
        
        print(f"\nSummary ({summary_result['num_documents']} documents):")
        print(summary_result['summary'])
    except Exception as e:
        print(f"Error generating summary: {e}")
    
    # Example 4: Query-based Summarization
    print("\n=== Example 4: Query-based Summarization ===")
    try:
        query = "machine learning"  # Replace with your query
        print(f"Summarizing documents related to: {query}")
        
        query_summary = summarizer.summarize_query_results(
            query=query,
            k=5
        )
        
        print(f"\nSummary ({query_summary['num_documents']} relevant documents):")
        print(query_summary['summary'])
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Examples Complete ===")


if __name__ == "__main__":
    main()

