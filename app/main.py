from fastapi import FastAPI
from pydantic import BaseModel
from app.pipeline import process_data
from app.vectorization import retrieve_and_generate_summary

app = FastAPI()

# Query request model
class QueryRequest(BaseModel):
    query: str

@app.post("/query/")
async def query_data(request: QueryRequest):
    query = request.query
    result = retrieve_and_generate_summary(query)
    return {"summary": result}

@app.on_event("startup")
async def startup():
    # Initialize Pinecone and load data
    process_data()
