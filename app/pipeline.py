import json
import pandas as pd
import hashlib
from app.vectorization import model
import pinecone
from app.vectorization import embed_data
from app.config import PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME,PINECONE_HOST

# Setup Pinecone connection
index = pinecone.Index(api_key=PINECONE_API_KEY, index_name=PINECONE_INDEX_NAME, host=PINECONE_HOST)

# Define the Pinecone index name and setup

# def process_data():
#     # Load and preprocess the data
#     with open("data/sample_data.json", "r") as file:
#         data = json.load(file)

#     # Clean and preprocess the reviews and other fields
#     df = pd.DataFrame(data)
#     df['review'] = df['review'].fillna('')  # Handle missing reviews
    
#     # Generate embeddings for reviews
#     embeddings = embed_data(df['review'].tolist())

#     # Insert into Pinecone
#     for i, embedding in enumerate(embeddings):
#         index.upsert([(str(df.iloc[i]['id']), embedding)])



def load_data(file_path):
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_embeddings_and_upsert(data):
    """Generate embeddings for the review chunks and upsert them with metadata into Pinecone."""
    for record in data:
        review = record["review"]
        
        # Split the review text into smaller chunks (you can define your own chunking strategy)
        chunks = review.split('.')  # This example splits by period. Adjust as necessary.
        
        for chunk in chunks:
            # Remove leading/trailing spaces and ensure each chunk has some content
            chunk = chunk.strip()
            if not chunk:
                continue

            # Create a unique hash for this chunk using SHA-256
            hash_value = hashlib.sha256(chunk.encode()).hexdigest()

            # Generate the embedding for the chunk
            chunk_embedding = model.encode(chunk).tolist()

            # Create the metadata for the chunk (this includes the original record info)
            metadata = {
                "id": record["id"],  # Original record ID
                "name": record["name"],
                "email": record["email"],
                "purchase_date": record["purchase_date"],
                "review_chunk": chunk,
                "tags": record["tags"]
            }

            # Upsert the chunk embedding and its metadata to Pinecone
            index.upsert(vectors=[{"id": hash_value, "values": chunk_embedding, "metadata": metadata}])

            print(f"Upserted chunk: {hash_value}")

def process_data():
    # Load the data from the JSON file
    data = load_data("data/sample_data.json")
    
    # Generate embeddings and upsert them to Pinecone
    generate_embeddings_and_upsert(data)

# if __name__ == "__main__":
#     main()