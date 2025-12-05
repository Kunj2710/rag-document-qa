# RAG System - Document Q&A and Summarization

A comprehensive Retrieval-Augmented Generation (RAG) system built with LangChain that enables document processing, question answering, text summarization, and output walkthroughs. The system supports multiple data sources, vector stores, and both cloud-based and local LLM models.

## Features

- **📄 Document Processing**: Load documents from PDFs, text files, URL, Wikipedia, ArXiv, etc.
- **❓ Question Answering**: Ask questions about your documents with context-aware responses
- **📝 Text Summarization**: Generate summaries using map-reduce, stuff, or refine methods
- **🔍 Output Walkthrough**: Get step-by-step explanations of how answers are generated
- **💾 Vector Store Support**: Multiple vector store backends (ChromaDB, FAISS, ObjectBox)
- **🌐 Interactive UI**: Streamlit-based web interface for easy interaction
- **🚀 REST API**: FastAPI backend with LangServe integration for programmatic access
- **🔌 Flexible LLM Integration**: Support for OpenAI, Groq (GPT-4o), and local models

## Architecture

### RAG Pipeline Components

1. **Data Retrieval**: Fetch data from web sources, local files, or APIs using LangChain's loaders
2. **Document Processing**: Break down documents into smaller chunks using text splitters
3. **Vector Embeddings**: Convert document chunks into high-dimensional vectors using OpenAI embeddings or sentence transformers
4. **Query Processing**: Retrieve relevant document chunks and use LLMs to generate context-based answers
5. **Interactive UI**: Streamlit interface for document uploads, querying, and result visualization
6. **API Service**: FastAPI backend for programmatic access with SSE support

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
```bash
cd rag-document-qa
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
   - Copy the example environment file (if available) or create a `.env` file:
   ```bash
   # Create .env file
   touch .env
   ```
   
   - Add your API keys to `.env`:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   
   VECTOR_STORE_TYPE=chromadb
   PERSIST_DIRECTORY=./vector_store
   
   LLM_MODEL=gpt-4o
   EMBEDDING_MODEL=text-embedding-3-small
   
   CHUNK_SIZE=1000
   CHUNK_OVERLAP=200
   ```

## Usage

### Streamlit UI (Recommended for Interactive Use)

Launch the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`.

#### Using the Streamlit UI:

1. **Upload Documents**:
   - Click "Browse files" in the sidebar
   - Select PDF or TXT files
   - Click "Load Documents"

2. **Ask Questions**:
   - Navigate to the "Question Answering" tab
   - Enter your question
   - Adjust the number of documents to retrieve
   - Click "Get Answer"
   - View the answer, source documents, and optional walkthrough

3. **Generate Summaries**:
   - Navigate to the "Summarization" tab
   - Choose summary type (map_reduce, stuff, or refine)
   - Click "Summarize All Documents" or use query-based summarization

4. **View Walkthroughs**:
   - Navigate to the "Output Walkthrough" tab
   - Enter a question
   - Click "Generate Walkthrough" to see step-by-step explanations

### FastAPI Backend (For Programmatic Access)

Start the API server:

```bash
python api_server.py
```

Or using uvicorn directly:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. API documentation is available at `http://localhost:8000/docs`.

#### API Endpoints:

- **POST `/upload`**: Upload and process documents
  ```bash
  curl -X POST "http://localhost:8000/upload" \
    -F "files=@document.pdf" \
    -F "files=@document2.txt"
  ```

- **POST `/qa`**: Ask questions
  ```bash
  curl -X POST "http://localhost:8000/qa" \
    -H "Content-Type: application/json" \
    -d '{
      "question": "What is the main topic?",
      "k": 4,
      "include_walkthrough": true
    }'
  ```

- **POST `/summarize`**: Generate summaries
  ```bash
  curl -X POST "http://localhost:8000/summarize" \
    -H "Content-Type: application/json" \
    -d '{
      "summary_type": "map_reduce",
      "query": "machine learning"
    }'
  ```

- **POST `/walkthrough`**: Generate walkthrough
  ```bash
  curl -X POST "http://localhost:8000/walkthrough" \
    -H "Content-Type: application/json" \
    -d '{
      "question": "What are the key findings?",
      "k": 4
    }'
  ```

- **GET `/stream/qa`**: Stream QA responses (SSE)
  ```bash
  curl "http://localhost:8000/stream/qa?question=What%20is%20this%20about?&k=4"
  ```

### Python API (Direct Usage)

```python
from rag_pipeline import RAGPipeline
from query_processor import QueryProcessor

# Initialize pipeline
pipeline = RAGPipeline()

# Load documents
pipeline.ingest_documents(
    source="path/to/document.pdf",
    source_type="file"
)

# Or load from multiple sources
pipeline.ingest_documents(
    source=["doc1.pdf", "doc2.txt"],
    source_type="file"
)

# Initialize query processor
query_processor = QueryProcessor(pipeline)

# Ask questions
result = query_processor.process_query(
    question="What is the main topic?",
    include_walkthrough=True,
    k=4
)

print(result["answer"])
print(result["walkthrough"])

# Generate summaries
from query_processor import DocumentSummarizer, LLMManager

summarizer = DocumentSummarizer(pipeline, LLMManager())
summary = summarizer.summarize_documents(summary_type="map_reduce")
print(summary["summary"])
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM | Required |
| `GROQ_API_KEY` | Groq API key for GPT-4o access | Optional |
| `VECTOR_STORE_TYPE` | Vector store backend (chromadb, faiss, objectbox) | chromadb |
| `PERSIST_DIRECTORY` | Directory to persist vector store | ./vector_store |
| `LLM_MODEL` | LLM model name | gpt-4o |
| `EMBEDDING_MODEL` | Embedding model name | text-embedding-3-small |
| `CHUNK_SIZE` | Document chunk size | 1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 |
| `API_HOST` | API server host | 0.0.0.0 |
| `API_PORT` | API server port | 8000 |

### Vector Store Options

- **ChromaDB** (default): Persistent, easy to use, good for development
- **FAISS**: Fast similarity search, good for large-scale applications
- **ObjectBox**: High-performance vector operations

### LLM Models

- **OpenAI**: GPT-4o, GPT-4-turbo, GPT-3.5-turbo
- **Groq**: Fast inference for GPT-4o
- **Local Models**: Can be configured with compatible LangChain integrations

## Project Structure

```
TermProject_CS688/
├── config.py                 # Configuration management
├── rag_pipeline.py           # Core RAG pipeline (document processing, vector store)
├── query_processor.py        # Query processing (QA, summarization, walkthrough)
├── streamlit_app.py          # Streamlit UI application
├── api_server.py             # FastAPI backend server
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Dependencies

### Core Libraries
- **langchain**: Core RAG framework
- **langchain-openai**: OpenAI integrations
- **langchain-community**: Community tools and loaders
- **langserve**: LangChain service deployment

### Document Processing
- **pypdf**, **PyPDF2**: PDF processing
- **beautifulsoup4**: Web scraping
- **wikipedia**, **arxiv**: Online data sources

### Vector Stores
- **chromadb**: Vector database
- **faiss-cpu**: Similarity search
- **sentence-transformers**: Local embeddings
- **langchain-objectbox**: ObjectBox integration

### Web Framework
- **streamlit**: Interactive UI
- **fastapi**: REST API
- **uvicorn**: ASGI server
- **sse-starlette**: Server-sent events

## How It Works

### Document Processing Flow

1. **Document Loading**: Documents are loaded from various sources (files, URLs, APIs)
2. **Text Splitting**: Large documents are split into smaller chunks for efficient processing
3. **Embedding Generation**: Each chunk is converted to a vector embedding
4. **Vector Storage**: Embeddings are stored in a vector database for fast retrieval

### Query Processing Flow

1. **Query Embedding**: User query is converted to a vector embedding
2. **Similarity Search**: Most relevant document chunks are retrieved
3. **Context Assembly**: Retrieved chunks are assembled as context
4. **LLM Generation**: LLM generates answer based on context and query
5. **Walkthrough Generation**: Optional step-by-step explanation of the process

class CustomLoader(BaseLoader):
    def load(self):
        # Your custom loading logic
        pass
```

### Custom Embeddings

Replace embeddings by modifying `VectorStoreManager._initialize_embeddings()`:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

self.embeddings = HuggingFaceEmbeddings(
    model_name='your-model-name'
)
```

### Custom Prompts

Modify prompts in `query_processor.py` to customize answer generation:

```python
prompt_template = """Your custom prompt here...
Context: {context}
Question: {question}
"""
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- Vector store support from ChromaDB, FAISS, and ObjectBox
- UI powered by Streamlit
- API framework: FastAPI

## Support

For issues, questions, or contributions, please open an issue on the project repository.
