# AI Operations Assistant

A multi-agent AI assistant that plans, executes, and verifies tasks using LLMs and real tools.

## Structure
- `agents/`: Core agent logic (Planner, Executor, Verifier).
- `tools/`: Tool definitions (GitHub, Weather).
- `llm/`: LLM client abstraction.
- `main.py`: Entry point.

## Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (for local LLM execution)

## Setup Guide

### 1. Install & Configure Ollama (Local LLM)
To run this assistant locally without API fees, we use **Ollama**.

**Option A: Download via Website (Recommended)**
1. Go to [ollama.com/download](https://ollama.com/download).
2. Download the macOS version.
3. Install and run the application.

**Option B: Install via Homebrew**
```bash
brew install ollama
brew services start ollama
```

**Initialize the Model**
Once Ollama is installed, open a terminal and run:
```bash
# This downloads the Llama 3 model (approx 4.7GB) and starts the server
ollama run llama3
```
*Keep this terminal open or ensure the Ollama menu bar app is running.*

### 2. Project Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**:
   Copy the example env file:
   ```bash
   cp .env.example .env
   ```
   
   Ensure `.env` matches your setup:
   ```bash
   LLM_PROVIDER=local
   LLM_BASE_URL=http://localhost:11434/v1
   LLM_API_KEY=ollama
   LLM_MODEL=llama3
   ```

3. **Run the Assistant**:
   ```bash
   # Run from the parent directory of this folder
   python -m ai_ops_assistant.main
   ```

## Tests
Run the unit test from the parent directory:

```bash
python -m unittest -q ai_ops_assistant.verification_test
```

## Usage
Enter natural language queries at the prompt. 
- "What is the weather in Paris?"
- "Find me a python library for web scraping on GitHub."
- "Check the weather in New York and find a react tutorial on GitHub."
