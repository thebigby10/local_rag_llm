## generating embeddings
import nltk
nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def split_document(doc, chunk_size=5):
    sentences = sent_tokenize(doc)
    return [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

from transformers import BertTokenizer, BertModel
#import torch

def generate_embeddings(chunks):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return embeddings

import faiss
import numpy as np

def save_to_faiss(embeddings, chunks):
    index_file = "document_index.faiss"
    chunks_file = "text_chunks.json"

    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(embeddings[0].shape[1])
    index.add(np.vstack(embeddings))
    faiss.write_index(index, index_file)

    # Save text chunks to JSON
    if os.path.exists(chunks_file):
        # Load existing chunks and append new ones
        with open(chunks_file, "r") as file:
            existing_chunks = json.load(file)
    else:
        # Initialize if the file doesn't exist
        existing_chunks = []
    
    # Append new chunks
    existing_chunks.extend(chunks)
    
    with open(chunks_file, "w") as file:
        json.dump(existing_chunks, file)
    print(f"Text chunks saved to {chunks_file}")

def get_context_from_faiss(query_embedding):
    index_file = "document_index.faiss"
    chunks_file = "text_chunks.json"


    index = faiss.read_index("document_index.faiss")
    # Search for nearest neighbors
    distances, indices = index.search(query_embedding,k=5)
    
    # Load text chunks
    with open(chunks_file, "r") as file:
        chunks = json.load(file)
    
    # Fetch corresponding chunks
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    
    return relevant_chunks, distances[0]



## Embedding query
def embed_query(query):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def search_faiss(query_embedding):
    index = faiss.read_index("document_index.faiss")
    distances, indices = index.search(query_embedding, k=5)  # Top 5 results
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return relevant_chunks    

## querying gemini with context
import requests
import os
import json
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('gemini_api_key')

url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"



def query_gemini(relevant_chunks, query):
    prompt = f"Context:\n{relevant_chunks}\n\nQuestion: {query}"
    headers = {"Content-Type": "application/json"}
    payload = {
    "contents": [
        {
            "parts": [
                {"text": f"{prompt}"}
            ]
        }
    ]
}

    response = requests.post(
        f"{url}?key={api_key}",
        headers=headers,
        data=json.dumps(payload),
    )
    return response.json()

from flask import Flask, request, jsonify 

app = Flask(__name__)

@app.route('/context', methods=['POST'])
def add_context():
    request_data = request.get_json()
    if request_data is None:
        return jsonify({'error': 'No data received'})

    split_documents = split_document(request_data['context'])
    embeddings = generate_embeddings(split_documents)
    save_to_faiss(embeddings, split_documents)
    return jsonify({'status': 'Context added successfully'}), 200

@app.route('/query', methods=['GET'])
def get_response():
    request_data = request.get_json()
    if request_data is None:
        return jsonify({'error': 'No data received'}), 400
    query = request_data['query']
    query_embedding = embed_query(query) 
    data = get_context_from_faiss(query_embedding)
    print(data[0])
    response = query_gemini(data[0], query)

    return jsonify(response), 200

