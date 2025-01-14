RAG Movie Recommendation Chatbot
This repository contains the code for a Retrieval-Augmented Generation (RAG) movie recommendation chatbot built using a movie dataset from Kaggle. The bot provides personalized movie recommendations by leveraging a combination of Sentence Transformers for generating embeddings, Qdrant for vector storage and similarity search, and the Llava-v1.5-7B language model for generating dynamic responses.

Features
Personalized movie recommendations based on user preferences.
Retrieval-based and generative approach combining Sentence Embeddings and Language Models.
Efficient search using Qdrant for storing and querying movie data vectors.
Architecture
Language Model (LLM):
The system uses the llava-v1.5-7b-q4.llamafile model for generating responses.

Encoder:
Sentence embeddings are created using the SentenceTransformer('all-MiniLM-L6-v2') encoder, which is optimized for sentence-level encoding.

Vector Database (Qdrant):
Qdrant is used for storing and querying vectors. The vectors are created by combining embeddings of the movie overview, genres, and keywords.

Search and Filtering:

The model queries the vector database for relevant movie results based on cosine similarity.
Results are filtered with a threshold score of 0.5 to ensure relevance.
Setup Instructions
Requirements
Python 3.7 or later
Libraries:
qdrant-client
sentence-transformers
torch
transformers
numpy
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/movie-recommendation-chatbot.git
cd movie-recommendation-chatbot
Install the required libraries:

bash
Copy code
pip install -r requirements.txt
Dataset
The movie dataset is sourced from Kaggle. Make sure to download and place the dataset in the appropriate folder or specify the path in the code.

Setup Qdrant Vector Database
Create a vector collection in Qdrant:
python
Copy code
qdrant.recreate_collection(
    collection_name="movies",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE
    )
)
Upload the movie data (embedding movie overview, genres, and keywords):
python
Copy code
qdrant.upload_records(
    collection_name="movies",
    records=[
        models.Record(
            id=idx,
            vector=(encoder.encode(doc["overview"]) + encoder.encode(doc["genres"]) + encoder.encode(doc["keywords"])).tolist(),
            payload=doc
        ) for idx, doc in enumerate(data)
    ]
)
Using the Chatbot
Run the script to interact with the chatbot and get movie recommendations:

bash
Copy code
python chatbot.py
You can provide input queries, and the system will generate personalized movie recommendations based on your preferences.

Filtering Results
The results are filtered with a score threshold (default set to 0.5):

python
Copy code
search_results = [hit.payload for hit in hits if hit.payload.get('score', 0) > 0.5]
This ensures that only relevant recommendations are displayed.

Example
Given an input query such as "I want to watch a sci-fi movie with action", the chatbot will return a list of recommended movies based on similarities in their overview, genres, and keywords.

