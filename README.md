# Reading Assistant

A personal reading assistant powered by RAG (Retrieval-Augmented Generation) technology. Upload your PDF documents and interact with them using your preferred language model.

<div style="text-align: center; margin: 20px 0;">
  <img src="assets/imgs/V1 - Reading Assistant's First Page.png" alt="Reading Assistant Interface" style="max-width: 75%; height: auto;">
</div>

## Prerequisites

- Python 3.11 or higher
- UV package manager installed ([Install UV](https://github.com/astral-sh/uv))

## Quick Start

1. **Install UV** (if not already installed):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Configure Environment Variables**:
   - Copy `.env.example` to `.env`:

     ```bash
     cp .env.example .env
     ```

   - Edit `.env` based on your preference, for example for using the OpenAI model you must define the following variables:

       ```bash
       # Required: Choose your model type
       MODEL_TYPE=openai
       
       # Required: OpenAI configuration (if using OpenAI)
       OPENAI_API_KEY=your_openai_api_key_here
       
       # Required: Embedding model for document processing
       # This is used for converting documents into vectors for similarity search
       EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
       
       # Optional: Server configuration
       SERVER_NAME=127.0.0.1
       SERVER_PORT=7860
       ```

3. **Run the Application**:

   ```bash
   uv run main.py
   ```

   UV will automatically:
   - Create a virtual environment
   - Install all required dependencies
   - Load environment variables
   - Start the application

4. **Access the Application**:
   - Open your web browser
   - Navigate to `http://127.0.0.1:7860` (or your configured server address)

> Note
The application will use the environment variables you've configured to set up the appropriate model and connection settings. Make sure all required API keys and tokens are properly configured in your `.env` file before running the application.
