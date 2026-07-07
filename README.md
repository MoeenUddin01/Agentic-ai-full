# AgenticAI

A project exploring Langchain and AI agent implementations with various model integrations.

## Overview

This project demonstrates the use of Langchain for building AI agents and applications with support for multiple AI providers including OpenAI, Groq, and Google Generative AI.

## Features

- **Langchain Integration**: Core Langchain framework for building AI applications
- **Multiple Model Support**: Integration with OpenAI, Groq, Google GenAI, and OpenRouter
- **DeepAgents Framework**: Advanced agent library with backend management (StateBackend, FileSystemBackend)
- **Context Engineering**: System prompt design and context management strategies
- **Streaming Capabilities**: Real-time streaming of AI responses
- **Tool Integration**: Langchain tools for extending agent capabilities
- **Jupyter Notebooks**: Interactive notebooks for learning and experimentation

## Project Structure

```
AgenticAI/
├── Langchain/
│   ├── langchain_intro.ipynb    # Introduction to Langchain concepts
│   ├── model_integraion.ipynb   # Model integration examples
│   ├── streaming.ipynb           # Streaming response implementations
│   └── tools.ipynb               # Langchain tools usage
├── deepAgent/
│   └── notebooks/
│       ├── deepagent.ipynb           # DeepAgent implementation using deepagents library
│       ├── deepagent_fixed.ipynb     # Fixed version with corrected imports/syntax
│       ├── basic.ipynb               # Basic vs DeepAgent comparison notebook
│       ├── 3-backend.ipynb           # Backend comparison (StateBackend vs FileSystemBackend)
│       └── context-engineering.ipynb # System prompts and context management strategies
├── notes/
│   └── todo.txt                      # Example files created by agents
├── RAG/
│   ├── pdf_loader.ipynb          # RAG pipeline (data ingestion → vector DB → LLM)
│   └── pdf_loader2.ipynb         # Fixed & enhanced RAG pipeline
├── src/
│   ├── __init__.py
│   ├── core/                     # Global configurations and logging
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── security.py           # e.g., role-based access control
│   ├── ingestion/                # Document processing and storage
│   │   ├── loaders.py            # Extract text/tables from PDFs/Docs
│   │   ├── chunkers.py           # Structure-aware chunking rules
│   │   └── indexers.py           # Metadata handling and vector uploads
│   ├── retrieval/                # Information retrieval and ranking
│   │   ├── embedder.py           # Text to vector conversion
│   │   ├── vector_db.py          # Database clients (e.g., Pinecone, Milvus)
│   │   ├── hybrid_search.py      # Dense/sparse merging & MMR
│   │   └── rerank.py             # Cross-encoder re-rankers
│   ├── generation/               # LLM & Response generation
│   │   ├── prompts.py            # Prompt templates & context budgeting
│   │   ├── guardrails.py         # Response constraints and hallucination checks
│   │   └── llm_client.py         # Calls to hosted/local LLM endpoints
│   ├── evaluation/               # Retrieval quality & citation metrics
│   │   └── metrics.py
│   ├── data_loader.py
│   ├── vectorstore.py
│   ├── embedding.py
│   ├── chunking.py
│   └── search.py
├── main.py                       # Main application entry point
├── pyproject.toml                # Project dependencies and configuration
└── .env                          # Environment variables (API keys)
```

## Requirements

- Python 3.12 or higher
- uv (for dependency management)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AgenticAI
```

2. Install dependencies using uv:
```bash
uv sync
```

## Setup

1. Create a `.env` file in the project root with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
GOOGLE_API_KEY=your_google_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Usage

### Running the Main Application
```bash
python main.py
```

### Working with Jupyter Notebooks

Launch Jupyter to explore the interactive notebooks:
```bash
jupyter notebook
```

Then explore the available notebooks:

**Langchain basics:**
- **Langchain/langchain_intro.ipynb**: Learn Langchain basics
- **Langchain/model_integraion.ipynb**: See how to integrate different AI models
- **Langchain/streaming.ipynb**: Implement streaming responses
- **Langchain/tools.ipynb**: Use Langchain tools to extend agent capabilities

**DeepAgent:**
- **deepAgent/notebooks/deepagent_fixed.ipynb**: Create agents using the DeepAgent library with Groq and Tavily search
- **deepAgent/notebooks/basic.ipynb**: Compare basic LangChain agents vs DeepAgent
- **deepAgent/notebooks/3-backend.ipynb**: Backend comparison - StateBackend (in-memory) vs FileSystemBackend (disk)
- **deepAgent/notebooks/context-engineering.ipynb**: System prompts and context management for agents

**RAG pipelines:**
- **RAG/pdf_loader2.ipynb**: Document loading, chunking, embedding, and retrieval-augmented generation

## Dependencies

- langchain>=1.0.0
- langchain-openai>=1.0.0
- langchain-groq>=0.0.1
- langchain-google-genai>=0.0.1
- langchain_community>=0.0.1
- deepagents>=0.6.0
- python-dotenv>=1.0.0
- ipykernel>=6.0.0
- jupyter>=1.0.0
- notebook>=7.0.0
- ipywidgets>=8.0.0

## License

Add your license information here.
