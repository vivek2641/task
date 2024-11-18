from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pinecone
from app.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,PINECONE_HOST

# Load Sentence Transformer and Summarization Model
model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Initialize Pinecone client
index = pinecone.Index(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME, host=PINECONE_HOST)


# Function to create embeddings for text
def embed_data(texts):
    embeddings = model.encode(texts)
    return embeddings

# Function to query Pinecone and generate summary
def retrieve_and_generate_summary(query):
    # Step 1: Generate embedding for the query
    embedding = model.encode(query).tolist()

    # Step 2: Query Pinecone for similar items
    result = index.query(vector=[embedding], top_k=5, include_metadata=True)

    # Step 3: Combine retrieved reviews into one string
    combined_text = " ".join([item['metadata']['review'] for item in result['matches']])

    # Step 4: Generate summary using summarizer
    summary = summarizer(combined_text, max_length=50, min_length=25, do_sample=False)
    return summary[0]["summary_text"]
