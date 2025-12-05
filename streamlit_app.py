"""
Streamlit UI Application for RAG System
Provides interactive interface for document upload, QA, and summarization.
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from typing import Optional

from rag_pipeline import RAGPipeline
from query_processor import QueryProcessor, DocumentSummarizer, LLMManager
from config import Config
from audio_generator import AudioGenerator


# Page configuration
st.set_page_config(
    page_title="RAG System - Document Q&A",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "query_processor" not in st.session_state:
    st.session_state.query_processor = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


def initialize_pipeline():
    """Initialize RAG pipeline and query processor."""
    if st.session_state.rag_pipeline is None:
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.query_processor = QueryProcessor(st.session_state.rag_pipeline)


def load_documents_from_upload(uploaded_files):
    """Load documents from uploaded files."""
    if not uploaded_files:
        return False
    
    initialize_pipeline()
    
    # Save uploaded files temporarily
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    
    try:
        with st.spinner("Processing documents..."):
            # Process each file and add to existing vector store
            # This ensures all documents are added incrementally instead of overwriting
            for file_path in file_paths:
                st.session_state.rag_pipeline.ingest_documents(
                    source=file_path,
                    source_type="file",
                    add_to_existing=True  # Add to existing store instead of overwriting
                )
            
            st.session_state.documents_loaded = True
            return True
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return False


def main():
    """Main Streamlit application."""
    st.title("📚 RAG System - Document Q&A and Summarization")
    st.markdown("Upload documents and ask questions or get summaries using RAG (Retrieval-Augmented Generation)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
               
        st.subheader("🌐 Online Sources")

        # Make sure pipeline is initialized before using it
        def ensure_pipeline():
            if "rag_pipeline" not in st.session_state or st.session_state.rag_pipeline is None:
                initialize_pipeline()

        # 1) Load from Web URL
        web_url = st.text_input("Web URL (https://...)")
        if st.button("Load from Web URL"):
            if not web_url:
                st.warning("Please enter a valid URL.")
            else:
                try:
                    ensure_pipeline()
                    # Ingest web page into vector store
                    st.session_state.rag_pipeline.ingest_documents(
                        source=web_url,
                        source_type="url"   # your backend should route this to load_web_url()
                    )
                    st.session_state.documents_loaded = True
                    st.success("Successfully loaded document from web URL!")
                except Exception as e:
                    st.error(f"Error loading web URL: {str(e)}")

        # 2) Load from Wikipedia
        wiki_query = st.text_input("Wikipedia topic (e.g., 'Machine learning')")
        if st.button("Load from Wikipedia"):
            if not wiki_query:
                st.warning("Please enter a Wikipedia topic.")
            else:
                try:
                    ensure_pipeline()
                    st.session_state.rag_pipeline.ingest_documents(
                        source=wiki_query,
                        source_type="wikipedia"  # backend should call load_wikipedia()
                    )
                    st.session_state.documents_loaded = True
                    st.success(f"Successfully loaded Wikipedia article: {wiki_query}")
                except Exception as e:
                    st.error(f"Error loading Wikipedia article: {str(e)}")

        # 3) Load from arXiv
        arxiv_query = st.text_input("arXiv query or ID (e.g., 'transformers', '1706.03762')")
        if st.button("Load from arXiv"):
            if not arxiv_query:
                st.warning("Please enter an arXiv query or paper ID.")
            else:
                try:
                    ensure_pipeline()
                    st.session_state.rag_pipeline.ingest_documents(
                        source=arxiv_query,
                        source_type="arxiv"  # backend should call load_arxiv()
                    )
                    st.session_state.documents_loaded = True
                    st.success(f"Successfully loaded arXiv documents for: {arxiv_query}")
                except Exception as e:
                    st.error(f"Error loading arXiv documents: {str(e)}")

        
        # Document upload
        st.subheader("📄 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Load Documents"):
            if load_documents_from_upload(uploaded_files):
                st.success(f"Successfully loaded {len(uploaded_files)} document(s)!")
        
        # Load existing vector store
        st.subheader("💾 Vector Store")
        if st.button("Load Existing Vector Store"):
            initialize_pipeline()
            try:
                st.session_state.rag_pipeline.load_existing_store()
                st.session_state.documents_loaded = True
                st.success("Vector store loaded successfully!")
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
        
        # Clear vector store
        st.subheader("🗑️ Clear Data")
        if st.button("Clear Session (Memory Only)"):
            st.session_state.rag_pipeline = None
            st.session_state.query_processor = None
            st.session_state.documents_loaded = False
            st.success("Session cleared!")
            st.rerun()
        
        if st.button("Clear Vector Store (Delete All Documents)", type="secondary"):
            if st.session_state.rag_pipeline:
                try:
                    st.session_state.rag_pipeline.clear_vector_store()
                    st.session_state.rag_pipeline = None
                    st.session_state.query_processor = None
                    st.session_state.documents_loaded = False
                    st.success("Vector store cleared! All documents removed.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing vector store: {str(e)}")
            else:
                st.warning("No vector store to clear.")
        
        # Status
        st.subheader("📊 Status")
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded")
        else:
            st.warning("⚠️ No documents loaded")
    
    # Main content area
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload documents in the sidebar to get started.")
        st.markdown("""
        ### Features:
        - **Document Upload**: Upload PDF, TXT, URL, Wikipedia, ArXiv files
        - **Question Answering**: Ask questions about your documents
        - **Text Summarization**: Get summaries of your documents
        - **Output Walkthrough**: See step-by-step explanations of answers
        """)
    else:
        # Tabs for different functionalities
        tab1, tab2, tab3 = st.tabs(["❓ Question Answering", "📝 Summarization", "🔍 Output Walkthrough"])
        
        # Tab 1: Question Answering
        with tab1:
            st.header("Ask Questions About Your Documents")
            
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the main topic of this document?"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                num_docs = st.slider("Number of documents to retrieve", 1, 10, 4)
            with col2:
                include_walkthrough = st.checkbox("Include walkthrough", value=True)
            
            if st.button("Get Answer", type="primary"):
                if question:
                    with st.spinner("Processing your question..."):
                        try:
                            result = st.session_state.query_processor.process_query(
                                question=question,
                                include_walkthrough=include_walkthrough,
                                k=num_docs
                            )
                            
                            # Display answer
                            st.subheader("Answer")
                            st.write(result["answer"])
                            
                            # Audio generation for answer
                            st.subheader("🔊 Audio Output")
                            col_audio1, col_audio2 = st.columns(2)
                            with col_audio1:
                                generate_audio = st.button("Generate Audio", key="qa_audio")
                            with col_audio2:
                                audio_engine = st.selectbox("TTS Engine", ["gtts", "pyttsx3"], key="qa_engine")
                            
                            if generate_audio:
                                with st.spinner("Generating audio..."):
                                    try:
                                        audio_gen = AudioGenerator(engine=audio_engine)
                                        audio_bytes = audio_gen.generate_audio(result["answer"])
                                        
                                        # Determine file extension
                                        ext = ".mp3" if audio_engine == "gtts" else ".wav"
                                        
                                        # Create download button
                                        st.audio(audio_bytes, format=f"audio/{ext[1:]}")
                                        st.download_button(
                                            label="Download Audio",
                                            data=audio_bytes,
                                            file_name=f"answer_{hash(result['question']) % 10000}{ext}",
                                            mime=f"audio/{ext[1:]}"
                                        )
                                    except Exception as e:
                                        st.error(f"Error generating audio: {str(e)}")
                            
                            # Display sources
                            with st.expander(f"📚 Source Documents ({result['num_sources']})"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(doc["content"][:500] + "..." if len(doc["content"]) > 500 else doc["content"])
                                    if doc.get("metadata"):
                                        st.caption(f"Metadata: {doc['metadata']}")
                                    st.divider()
                            
                            # Display walkthrough if requested
                            if include_walkthrough and "walkthrough" in result:
                                with st.expander("🔍 Step-by-Step Walkthrough"):
                                    st.write(result["walkthrough"])
                                    
                                    if "steps" in result:
                                        st.markdown("### Key Steps:")
                                        for step in result["steps"]:
                                            st.markdown(f"- {step}")
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
                else:
                    st.warning("Please enter a question.")
        
        # Tab 2: Summarization
        with tab2:
            st.header("Document Summarization")
            
            summary_type = st.selectbox(
                "Summary Type",
                ["map_reduce", "stuff", "refine"],
                help="map_reduce: Best for large documents\nstuff: Best for small documents\nrefine: Most detailed"
            )
            
            # Add options for filtering
            st.subheader("Filter Options")
            summarize_option = st.radio(
                "What to summarize:",
                ["All Documents", "Most Recent Upload", "Specific Document"],
                help="Choose which documents to include in the summary"
            )
            
            filter_source = None
            if summarize_option == "Specific Document":
                # Get list of unique source files from vector store
                try:
                    all_docs = st.session_state.rag_pipeline.vector_store.similarity_search("", k=1000)
                    source_files = set()
                    for doc in all_docs:
                        if doc.metadata and 'source_file' in doc.metadata:
                            source_files.add(doc.metadata['source_file'])
                    if source_files:
                        filter_source = st.selectbox(
                            "Select document to summarize:",
                            sorted(source_files)
                        )
                    else:
                        st.info("No documents with source file metadata found.")
                except:
                    st.warning("Could not retrieve document list.")
            
            col1, col2 = st.columns(2)
            with col1:
                summarize_all = st.button("Generate Summary", type="primary")
            with col2:
                query_summary = st.button("Summarize by Query")
            
            if summarize_all:
                with st.spinner("Generating summary..."):
                    try:
                        summarizer = DocumentSummarizer(
                            st.session_state.rag_pipeline,
                            LLMManager()
                        )
                        
                        # Apply filters based on selection
                        if summarize_option == "Most Recent Upload":
                            result = summarizer.summarize_documents(
                                summary_type=summary_type,
                                only_recent=True,
                                recent_threshold_seconds=3600.0  # Last hour
                            )
                        elif summarize_option == "Specific Document" and filter_source:
                            result = summarizer.summarize_documents(
                                summary_type=summary_type,
                                filter_by_source=filter_source
                            )
                        else:
                            result = summarizer.summarize_documents(summary_type=summary_type)
                        
                        st.subheader("Summary")
                        st.write(result["summary"])
                        st.caption(f"Based on {result['num_documents']} document chunks")
                        
                        if summarize_option != "All Documents":
                            st.info(f"Filtered: {summarize_option}")
                        
                        # Audio generation for summary
                        st.subheader("🔊 Audio Output")
                        col_audio1, col_audio2 = st.columns(2)
                        with col_audio1:
                            generate_audio_summary = st.button("Generate Audio", key="summary_audio")
                        with col_audio2:
                            audio_engine_summary = st.selectbox("TTS Engine", ["gtts", "pyttsx3"], key="summary_engine")
                        
                        if generate_audio_summary:
                            with st.spinner("Generating audio..."):
                                try:
                                    audio_gen = AudioGenerator(engine=audio_engine_summary)
                                    audio_bytes = audio_gen.generate_audio(result["summary"])
                                    
                                    # Determine file extension
                                    ext = ".mp3" if audio_engine_summary == "gtts" else ".wav"
                                    
                                    # Create download button
                                    st.audio(audio_bytes, format=f"audio/{ext[1:]}")
                                    st.download_button(
                                        label="Download Audio",
                                        data=audio_bytes,
                                        file_name=f"summary_{hash(str(result.get('query', 'all'))) % 10000}{ext}",
                                        mime=f"audio/{ext[1:]}"
                                    )
                                except Exception as e:
                                    st.error(f"Error generating audio: {str(e)}")
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
            
            if query_summary:
                query = st.text_input(
                    "Enter query to find relevant documents:",
                    key="summary_query",
                    placeholder="e.g., machine learning"
                )
                num_docs_summary = st.slider(
                    "Number of documents to summarize",
                    1, 20, 10,
                    key="summary_num_docs"
                )
                
                if st.button("Generate Query-Based Summary", key="gen_query_summary"):
                    if query:
                        with st.spinner("Generating summary..."):
                            try:
                                summarizer = DocumentSummarizer(
                                    st.session_state.rag_pipeline,
                                    LLMManager()
                                )
                                result = summarizer.summarize_query_results(
                                    query=query,
                                    k=num_docs_summary
                                )
                                
                                st.subheader("Summary")
                                st.write(result["summary"])
                                st.caption(f"Based on {result['num_documents']} relevant documents")
                                
                                # Audio generation for query-based summary
                                st.subheader("🔊 Audio Output")
                                col_audio1, col_audio2 = st.columns(2)
                                with col_audio1:
                                    generate_audio_query = st.button("Generate Audio", key="query_summary_audio")
                                with col_audio2:
                                    audio_engine_query = st.selectbox("TTS Engine", ["gtts", "pyttsx3"], key="query_summary_engine")
                                
                                if generate_audio_query:
                                    with st.spinner("Generating audio..."):
                                        try:
                                            audio_gen = AudioGenerator(engine=audio_engine_query)
                                            audio_bytes = audio_gen.generate_audio(result["summary"])
                                            
                                            # Determine file extension
                                            ext = ".mp3" if audio_engine_query == "gtts" else ".wav"
                                            
                                            # Create download button
                                            st.audio(audio_bytes, format=f"audio/{ext[1:]}")
                                            st.download_button(
                                                label="Download Audio",
                                                data=audio_bytes,
                                                file_name=f"query_summary_{hash(query) % 10000}{ext}",
                                                mime=f"audio/{ext[1:]}"
                                            )
                                        except Exception as e:
                                            st.error(f"Error generating audio: {str(e)}")
                                
                                # Show source documents
                                with st.expander("Source Documents"):
                                    for i, doc in enumerate(result["source_documents"], 1):
                                        st.markdown(f"**Document {i}:**")
                                        st.text(doc["content"])
                                        if doc.get("metadata"):
                                            st.caption(f"Metadata: {doc['metadata']}")
                                        st.divider()
                            except Exception as e:
                                st.error(f"Error generating summary: {str(e)}")
                    else:
                        st.warning("Please enter a query.")
        
        # Tab 3: Output Walkthrough
        with tab3:
            st.header("Output Walkthrough Generator")
            st.markdown("Generate detailed explanations of how answers are derived from documents.")
            
            walkthrough_question = st.text_input(
                "Enter a question:",
                key="walkthrough_q",
                placeholder="e.g., What are the key findings?"
            )
            
            if st.button("Generate Walkthrough", type="primary"):
                if walkthrough_question:
                    with st.spinner("Generating walkthrough..."):
                        try:
                            # First get the answer
                            qa_result = st.session_state.query_processor.qa.answer_with_context(
                                walkthrough_question,
                                k=num_docs
                            )
                            
                            # Generate walkthrough
                            walkthrough_result = st.session_state.query_processor.walkthrough.generate_walkthrough(
                                question=walkthrough_question,
                                answer=qa_result["answer"],
                                source_documents=qa_result["source_documents"]
                            )
                            
                            st.subheader("Question")
                            st.write(walkthrough_result["question"])
                            
                            st.subheader("Answer")
                            st.write(walkthrough_result["answer"])
                            
                            st.subheader("Step-by-Step Walkthrough")
                            st.write(walkthrough_result["walkthrough"])
                            
                            if walkthrough_result.get("steps"):
                                st.markdown("### Key Steps:")
                                for step in walkthrough_result["steps"]:
                                    st.markdown(f"- {step}")
                            
                            st.subheader("Source Information")
                            st.caption(f"Used {walkthrough_result['num_sources']} source documents")
                            
                        except Exception as e:
                            st.error(f"Error generating walkthrough: {str(e)}")
                else:
                    st.warning("Please enter a question.")


if __name__ == "__main__":
    # Validate configuration
    Config.validate()
    main()

