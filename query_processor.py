"""
Query Processing Module
Handles question answering, summarization, and output walkthrough.
"""
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
try:
    from langchain_groq import ChatGroq
except ImportError:
    ChatGroq = None
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Use LCEL (LangChain Expression Language) for building chains
# Note: Old chains API is deprecated, using LCEL instead

from config import Config
from rag_pipeline import RAGPipeline


class LLMManager:
    """Manages LLM initialization and configuration."""
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.7):
        self.model_name = model_name or Config.LLM_MODEL
        self.temperature = temperature
        self.llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            # Try Groq first (for GPT-4o)
            if ChatGroq and Config.GROQ_API_KEY and "gpt-4o" in self.model_name.lower():
                self.llm = ChatGroq(
                    model_name=self.model_name,
                    groq_api_key=Config.GROQ_API_KEY,
                    temperature=self.temperature
                )
            # Try OpenAI
            elif Config.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model_name=self.model_name,
                    openai_api_key=Config.OPENAI_API_KEY,
                    temperature=self.temperature
                )
            else:
                raise ValueError("No API key available for LLM")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            # Fallback to OpenAI if available
            if Config.OPENAI_API_KEY:
                self.llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    openai_api_key=Config.OPENAI_API_KEY,
                    temperature=self.temperature
                )
            else:
                raise ValueError("Unable to initialize LLM. Please set API keys.")
    
    def get_llm(self):
        """Get the initialized LLM."""
        return self.llm


class QuestionAnswering:
    """Handles question answering using RAG."""
    
    def __init__(self, rag_pipeline: RAGPipeline, llm_manager: LLMManager):
        self.rag_pipeline = rag_pipeline
        self.llm = llm_manager.get_llm()
        self.retriever = None
    
    def setup_retriever(self, k: int = 4, search_type: str = "similarity"):
        """Setup the retriever for QA."""
        self.retriever = self.rag_pipeline.get_retriever(k, search_type)
    
    def answer_question(
        self, 
        question: str, 
        return_source_documents: bool = True,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            question: The question to answer
            return_source_documents: Whether to return source documents
            k: Number of documents to retrieve
        
        Returns:
            Dictionary with answer and optionally source documents
        """
        # Always get a fresh retriever to ensure we're using the latest vector store
        retriever = self.rag_pipeline.get_retriever(k, "similarity")
        
        # Retrieve relevant documents for THIS specific question
        # In newer LangChain versions, retrievers are callable or use invoke()
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older API
            try:
                docs = retriever.get_relevant_documents(question)
            except AttributeError:
                # Try calling directly
                docs = retriever(question)
        
        # Debug: Log what we retrieved
        if not docs:
            return {
                "question": question,
                "answer": "No relevant documents found in the vector store to answer this question.",
                "source_documents": []
            }
        
        # Create prompt
        prompt_template = """Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Format context
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        response = {
            "question": question,
            "answer": answer,
        }
        
        if return_source_documents:
            response["source_documents"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        
        return response
    
    def answer_with_context(
        self,
        question: str,
        k: int = 4,
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Answer question with detailed context and explanation.
        
        Args:
            question: The question to answer
            k: Number of documents to retrieve
            include_explanation: Whether to include step-by-step explanation
        
        Returns:
            Dictionary with answer, sources, and explanation
        """
        # Always get a fresh retriever to ensure we're using the latest vector store
        retriever = self.rag_pipeline.get_retriever(k, "similarity")
        
        # Retrieve relevant documents for THIS specific question
        # In newer LangChain versions, retrievers are callable or use invoke()
        try:
            docs = retriever.invoke(question)
        except AttributeError:
            # Fallback for older API
            try:
                docs = retriever.get_relevant_documents(question)
            except AttributeError:
                # Try calling directly
                docs = retriever(question)
        
        # Debug: Check if we got documents
        if not docs:
            return {
                "question": question,
                "answer": "No relevant documents found in the vector store to answer this question.",
                "source_documents": [],
                "num_sources": 0
            }
        
        # Create prompt with context
        prompt_template = """You are a helpful assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context. If the context doesn't contain enough information, say so.
"""
        
        if include_explanation:
            prompt_template += "\nAlso provide a brief explanation of how you arrived at this answer."
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Format context
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        
        # Generate answer
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({"context": context, "question": question})
        
        return {
            "question": question,
            "answer": answer,
            "source_documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', None)
                }
                for doc in docs
            ],
            "num_sources": len(docs)
        }


class DocumentSummarizer:
    """Handles document summarization."""
    
    def __init__(self, rag_pipeline: RAGPipeline, llm_manager: LLMManager):
        self.rag_pipeline = rag_pipeline
        self.llm = llm_manager.get_llm()
    
    def summarize_documents(
        self,
        summary_type: str = "map_reduce",
        max_tokens: Optional[int] = None,
        filter_by_source: Optional[str] = None,
        only_recent: bool = False,
        recent_threshold_seconds: float = 3600.0
    ) -> Dict[str, Any]:
        """
        Summarize documents in the vector store.
        
        Args:
            summary_type: Type of summarization - 'map_reduce', 'stuff', or 'refine'
            max_tokens: Maximum tokens for summary
            filter_by_source: Filter documents by source_file name (e.g., 'document.pdf')
            only_recent: If True, only summarize documents uploaded recently
            recent_threshold_seconds: Time threshold in seconds for 'recent' documents
        
        Returns:
            Dictionary with summary and metadata
        """
        import time
        
        # Get all documents from vector store
        if self.rag_pipeline.vector_store is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")
        
        # Retrieve all documents (you may want to limit this for large stores)
        all_docs = self.rag_pipeline.vector_store.similarity_search("", k=1000)
        
        # Filter documents if requested
        if filter_by_source or only_recent:
            filtered_docs = []
            current_time = time.time()
            
            for doc in all_docs:
                include = True
                
                # Filter by source file name
                if filter_by_source:
                    source_file = doc.metadata.get('source_file', '') if doc.metadata else ''
                    if filter_by_source not in source_file:
                        include = False
                
                # Filter by recent uploads
                if only_recent and include:
                    upload_time = doc.metadata.get('upload_timestamp', 0) if doc.metadata else 0
                    if current_time - upload_time > recent_threshold_seconds:
                        include = False
                
                if include:
                    filtered_docs.append(doc)
            
            all_docs = filtered_docs
        
        if not all_docs:
            return {"summary": "No documents found in the vector store.", "num_documents": 0}
        
        # Create summarization using LCEL
        # For map_reduce, we'll use a simpler approach
        if summary_type == "stuff":
            # Stuff all documents into one prompt
            combined_text = "\n\n".join([doc.page_content for doc in all_docs])
            prompt_template = """Summarize the following documents. Provide a comprehensive summary that captures the main points and key information.

Documents:
{text}

Summary:"""
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm | StrOutputParser()
            summary_text = chain.invoke({"text": combined_text})
        else:
            # For map_reduce and refine, use a simpler approach
            # Split into chunks and summarize each, then combine
            chunk_size = 5000  # Approximate chunk size
            chunks = []
            current_chunk = []
            current_length = 0
            
            for doc in all_docs:
                doc_text = doc.page_content
                if current_length + len(doc_text) > chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = [doc_text]
                    current_length = len(doc_text)
                else:
                    current_chunk.append(doc_text)
                    current_length += len(doc_text)
            
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks:
                prompt_template = """Summarize the following text:

{text}

Summary:"""
                prompt = ChatPromptTemplate.from_template(prompt_template)
                chain = prompt | self.llm | StrOutputParser()
                chunk_summary = chain.invoke({"text": chunk})
                chunk_summaries.append(chunk_summary)
            
            # Combine summaries
            if len(chunk_summaries) > 1:
                combined_summaries = "\n\n".join([f"Summary {i+1}:\n{s}" for i, s in enumerate(chunk_summaries)])
                final_prompt = ChatPromptTemplate.from_template("""Combine the following summaries into a single comprehensive summary:

{summaries}

Combined Summary:""")
                final_chain = final_prompt | self.llm | StrOutputParser()
                summary_text = final_chain.invoke({"summaries": combined_summaries})
            else:
                summary_text = chunk_summaries[0] if chunk_summaries else "No summary available."
        
        return {
            "summary": summary_text,
            "num_documents": len(all_docs),
            "summary_type": summary_type
        }
    
    def summarize_query_results(
        self,
        query: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """
        Summarize documents retrieved for a specific query.
        
        Args:
            query: Query to retrieve relevant documents
            k: Number of documents to retrieve and summarize
        
        Returns:
            Dictionary with summary and source information
        """
        retriever = self.rag_pipeline.get_retriever(k)
        # In newer LangChain versions, retrievers are callable or use invoke()
        try:
            docs = retriever.invoke(query)
        except AttributeError:
            # Fallback for older API
            try:
                docs = retriever.get_relevant_documents(query)
            except AttributeError:
                # Try calling directly
                docs = retriever(query)
        
        if not docs:
            return {"summary": "No relevant documents found.", "num_documents": 0}
        
        # Create summarization using LCEL
        combined_text = "\n\n".join([doc.page_content for doc in docs])
        prompt_template = """Summarize the following documents related to the query: {query}

Documents:
{text}

Summary:"""
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | StrOutputParser()
        summary_text = chain.invoke({"query": query, "text": combined_text})
        
        return {
            "query": query,
            "summary": summary_text,
            "num_documents": len(docs),
            "source_documents": [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in docs[:5]  # Return first 5 sources
            ]
        }


class OutputWalkthrough:
    """Provides step-by-step walkthrough of RAG outputs."""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm = llm_manager.get_llm()
    
    def generate_walkthrough(
        self,
        question: str,
        answer: str,
        source_documents: List[Dict[str, Any]],
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a step-by-step walkthrough of how the answer was generated.
        
        Args:
            question: The original question
            answer: The generated answer
            source_documents: List of source documents used
            include_reasoning: Whether to include reasoning steps
        
        Returns:
            Dictionary with walkthrough steps
        """
        walkthrough_prompt = """You are an AI assistant that explains how answers are generated from retrieved documents.

Question: {question}

Answer: {answer}

Source Documents Used:
{source_info}

Please provide a step-by-step walkthrough explaining:
1. How the relevant documents were identified
2. What key information was extracted from each document
3. How the information was synthesized to form the answer
4. Any limitations or uncertainties in the answer

Make it clear and educational.
"""
        
        source_info = "\n\n".join([
            f"Document {i+1}:\nContent: {doc.get('content', '')[:300]}...\nMetadata: {doc.get('metadata', {})}"
            for i, doc in enumerate(source_documents[:5])
        ])
        
        prompt = ChatPromptTemplate.from_template(walkthrough_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        walkthrough = chain.invoke({
            "question": question,
            "answer": answer,
            "source_info": source_info
        })
        
        return {
            "question": question,
            "answer": answer,
            "walkthrough": walkthrough,
            "num_sources": len(source_documents),
            "steps": self._extract_steps(walkthrough)
        }
    
    def _extract_steps(self, walkthrough: str) -> List[str]:
        """Extract numbered steps from walkthrough text."""
        lines = walkthrough.split('\n')
        steps = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                steps.append(line)
        return steps if steps else [walkthrough]


class QueryProcessor:
    """Main query processor that combines QA, summarization, and walkthrough."""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
        self.llm_manager = LLMManager()
        self.qa = QuestionAnswering(rag_pipeline, self.llm_manager)
        self.summarizer = DocumentSummarizer(rag_pipeline, self.llm_manager)
        self.walkthrough = OutputWalkthrough(self.llm_manager)
    
    def process_query(
        self,
        question: str,
        include_walkthrough: bool = True,
        k: int = 4
    ) -> Dict[str, Any]:
        """
        Process a query with full RAG pipeline including walkthrough.
        
        Args:
            question: The question to answer
            include_walkthrough: Whether to include step-by-step walkthrough
            k: Number of documents to retrieve
        
        Returns:
            Complete response with answer, sources, and walkthrough
        """
        # Get answer with context
        qa_result = self.qa.answer_with_context(question, k=k)
        
        result = {
            "question": qa_result["question"],
            "answer": qa_result["answer"],
            "source_documents": qa_result["source_documents"],
            "num_sources": qa_result["num_sources"]
        }
        
        # Add walkthrough if requested
        if include_walkthrough:
            walkthrough_result = self.walkthrough.generate_walkthrough(
                question=question,
                answer=qa_result["answer"],
                source_documents=qa_result["source_documents"]
            )
            result["walkthrough"] = walkthrough_result["walkthrough"]
            result["steps"] = walkthrough_result["steps"]
        
        return result

