# AgenticAI

A project exploring Langchain and AI agent implementations with various model integrations.

## Overview

This project demonstrates the use of Langchain for building AI agents and applications with support for multiple AI providers including OpenAI, Groq, and Google Generative AI.

## Features

- **Langchain Integration**: Core Langchain framework for building AI applications
- **Multiple Model Support**: Integration with OpenAI, Groq, and Google GenAI
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

Then navigate to the `Langchain/` directory to explore:
- **langchain_intro.ipynb**: Learn Langchain basics
- **model_integraion.ipynb**: See how to integrate different AI models
- **streaming.ipynb**: Implement streaming responses
- **tools.ipynb**: Use Langchain tools to extend agent capabilities

## Dependencies

- langchain>=1.0.0
- langchain-openai>=1.0.0
- langchain-groq>=0.0.1
- langchain-google-genai>=0.0.1
- langchain_community>=0.0.1
- python-dotenv>=1.0.0
- ipykernel>=6.0.0
- jupyter>=1.0.0
- notebook>=7.0.0
- ipywidgets>=8.0.0

## License

Add your license information here.
