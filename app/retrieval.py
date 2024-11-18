import pinecone
from app.vectorization import model
from transformers import pipeline
from app.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,PINECONE_HOST

# Initialize Pinecone client
index = pinecone.Index(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME, host=PINECONE_HOST)


# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def query_vector_db(embedding):
    """Query the Pinecone index to retrieve similar embeddings."""
    result = index.query(queries=[embedding], top_k=5, include_metadata=True)
    return result['matches']

def retrieve_and_generate_summary(query):
    """Retrieve relevant records and generate a summary."""
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()

    # Query Pinecone for similar items
    results = query_vector_db(query_embedding)

    # Combine retrieved reviews into one text
    combined_text = " ".join([item['metadata']['review'] for item in results])

    # Generate summary using the summarizer
    summary = summarizer(combined_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]
