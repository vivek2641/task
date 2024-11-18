# LLM Data Pipeline Project

This project provides an end-to-end pipeline for ingesting, processing, vectorizing, and querying text data using a vector database. The project structure facilitates retriever-augmented generation (RAG) using embeddings stored in a vector database.


## Setup Instructions
1. **Clone the repository**.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Running the Query API**:
   - Start the FastAPI server:
   ```bash
   uvicorn src.query_api:app --reload
   ```
   - Test the endpoint at `http://localhost:8000/query` with a JSON payload:
   ```json
   { "query": "Amazing product quality!" }
   ```

## Additional Information
- **config.py**: Stores connection settings for the vector database.
- **Logging**: Logs are saved in `pipeline.log` for tracking data flow and debugging.
