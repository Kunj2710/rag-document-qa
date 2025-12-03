"""
FastAPI Server for RAG System
Provides REST API endpoints for document processing, QA, and summarization.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import os
import tempfile
from sse_starlette.sse import EventSourceResponse
import json

from rag_pipeline import RAGPipeline
from query_processor import QueryProcessor, DocumentSummarizer, LLMManager
from config import Config
from audio_generator import AudioGenerator


app = FastAPI(
    title="RAG System API",
    description="REST API for RAG-based document Q&A and summarization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instances
rag_pipeline: Optional[RAGPipeline] = None
query_processor: Optional[QueryProcessor] = None


# Pydantic models for request/response
class QuestionRequest(BaseModel):
    question: str
    k: int = 4
    include_walkthrough: bool = True


class QuestionResponse(BaseModel):
    question: str
    answer: str
    source_documents: List[Dict[str, Any]]
    num_sources: int
    walkthrough: Optional[str] = None
    steps: Optional[List[str]] = None


class SummaryRequest(BaseModel):
    summary_type: str = "map_reduce"
    query: Optional[str] = None
    k: int = 10


class SummaryResponse(BaseModel):
    summary: str
    num_documents: int
    summary_type: str
    query: Optional[str] = None
    source_documents: Optional[List[Dict[str, Any]]] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global rag_pipeline, query_processor
    try:
        rag_pipeline = RAGPipeline()
        query_processor = QueryProcessor(rag_pipeline)
        # Try to load existing vector store
        try:
            rag_pipeline.load_existing_store()
        except:
            pass  # No existing store, that's okay
    except Exception as e:
        print(f"Warning: Could not initialize pipeline: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
            "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "qa": "/qa",
            "qa/audio": "/qa/audio",
            "summarize": "/summarize",
            "summarize/audio": "/summarize/audio",
            "walkthrough": "/walkthrough"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": rag_pipeline is not None,
        "vector_store_loaded": rag_pipeline.vector_store is not None if rag_pipeline else False
    }


@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process documents.
    
    Args:
        files: List of uploaded files (PDF or TXT)
    
    Returns:
        Status and document count
    """
    global rag_pipeline, query_processor
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Initialize pipeline if needed
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
        query_processor = QueryProcessor(rag_pipeline)
    
    # Save uploaded files temporarily
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    try:
        for uploaded_file in files:
            # Validate file type
            if not uploaded_file.filename.endswith(('.pdf', '.txt')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {uploaded_file.filename}"
                )
            
            file_path = os.path.join(temp_dir, uploaded_file.filename)
            with open(file_path, "wb") as f:
                content = await uploaded_file.read()
                f.write(content)
            file_paths.append(file_path)
        
        # Process documents
        for file_path in file_paths:
            rag_pipeline.ingest_documents(
                source=file_path,
                source_type="file"
            )
        
        return {
            "status": "success",
            "message": f"Successfully processed {len(files)} file(s)",
            "files_processed": len(files)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    
    finally:
        # Cleanup temp files
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass


@app.post("/qa", response_model=QuestionResponse)
async def question_answering(request: QuestionRequest):
    """
    Answer questions using RAG.
    
    Args:
        request: Question request with question, k, and include_walkthrough
    
    Returns:
        Answer with sources and optional walkthrough
    """
    if query_processor is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        result = query_processor.process_query(
            question=request.question,
            include_walkthrough=request.include_walkthrough,
            k=request.k
        )
        
        return QuestionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/qa/audio")
async def question_answering_audio(question: str, k: int = 4, engine: str = "gtts"):
    """
    Answer questions and return audio.
    
    Args:
        question: Question to answer
        k: Number of documents to retrieve
        engine: TTS engine ('gtts' or 'pyttsx3')
    
    Returns:
        Audio file of the answer
    """
    if query_processor is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        result = query_processor.process_query(
            question=question,
            include_walkthrough=False,
            k=k
        )
        
        # Generate audio
        audio_gen = AudioGenerator(engine=engine)
        audio_bytes = audio_gen.generate_audio(result["answer"])
        
        # Determine content type
        content_type = "audio/mpeg" if engine == "gtts" else "audio/wav"
        ext = ".mp3" if engine == "gtts" else ".wav"
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="answer{ext}"'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


@app.post("/summarize", response_model=SummaryResponse)
async def summarize_documents(request: SummaryRequest):
    """
    Summarize documents.
    
    Args:
        request: Summary request with type and optional query
    
    Returns:
        Summary of documents
    """
    if rag_pipeline is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        summarizer = DocumentSummarizer(rag_pipeline, LLMManager())
        
        if request.query:
            # Query-based summarization
            result = summarizer.summarize_query_results(
                query=request.query,
                k=request.k
            )
            return SummaryResponse(
                summary=result["summary"],
                num_documents=result["num_documents"],
                summary_type=request.summary_type,
                query=request.query,
                source_documents=result.get("source_documents")
            )
        else:
            # Summarize all documents
            result = summarizer.summarize_documents(
                summary_type=request.summary_type
            )
            return SummaryResponse(
                summary=result["summary"],
                num_documents=result["num_documents"],
                summary_type=result["summary_type"]
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.get("/summarize/audio")
async def summarize_audio(
    query: Optional[str] = None,
    summary_type: str = "map_reduce",
    k: int = 10,
    engine: str = "gtts"
):
    """
    Generate summary and return as audio.
    
    Args:
        query: Optional query to filter documents
        summary_type: Type of summarization
        k: Number of documents (if query provided)
        engine: TTS engine ('gtts' or 'pyttsx3')
    
    Returns:
        Audio file of the summary
    """
    if rag_pipeline is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        summarizer = DocumentSummarizer(rag_pipeline, LLMManager())
        
        if query:
            result = summarizer.summarize_query_results(query=query, k=k)
            summary_text = result["summary"]
        else:
            result = summarizer.summarize_documents(summary_type=summary_type)
            summary_text = result["summary"]
        
        # Generate audio
        audio_gen = AudioGenerator(engine=engine)
        audio_bytes = audio_gen.generate_audio(summary_text)
        
        # Determine content type
        content_type = "audio/mpeg" if engine == "gtts" else "audio/wav"
        ext = ".mp3" if engine == "gtts" else ".wav"
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={"Content-Disposition": f'attachment; filename="summary{ext}"'}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


@app.post("/walkthrough")
async def generate_walkthrough(request: QuestionRequest):
    """
    Generate step-by-step walkthrough of answer generation.
    
    Args:
        request: Question request
    
    Returns:
        Detailed walkthrough
    """
    if query_processor is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    try:
        # Get answer first
        qa_result = query_processor.qa.answer_with_context(
            request.question,
            k=request.k
        )
        
        # Generate walkthrough
        walkthrough_result = query_processor.walkthrough.generate_walkthrough(
            question=request.question,
            answer=qa_result["answer"],
            source_documents=qa_result["source_documents"]
        )
        
        return walkthrough_result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating walkthrough: {str(e)}")


@app.get("/stream/qa")
async def stream_qa(question: str, k: int = 4):
    """
    Stream QA responses using Server-Sent Events.
    
    Args:
        question: Question to answer
        k: Number of documents to retrieve
    """
    if query_processor is None or rag_pipeline.vector_store is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first."
        )
    
    async def event_generator():
        try:
            # Send initial message
            yield {
                "event": "status",
                "data": json.dumps({"message": "Processing question..."})
            }
            
            # Get answer
            result = query_processor.process_query(
                question=question,
                include_walkthrough=True,
                k=k
            )
            
            # Stream answer
            yield {
                "event": "answer",
                "data": json.dumps({"answer": result["answer"]})
            }
            
            # Stream sources
            yield {
                "event": "sources",
                "data": json.dumps({"sources": result["source_documents"]})
            }
            
            # Stream walkthrough
            if "walkthrough" in result:
                yield {
                    "event": "walkthrough",
                    "data": json.dumps({"walkthrough": result["walkthrough"]})
                }
            
            # Send completion
            yield {
                "event": "complete",
                "data": json.dumps({"status": "done"})
            }
        
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    Config.validate()
    uvicorn.run(
        "api_server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True
    )

