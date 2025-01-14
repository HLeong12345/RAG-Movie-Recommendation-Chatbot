# RAG Movie Recommendation Chatbot

This repository contains the code for a **Retrieval-Augmented Generation (RAG)** movie recommendation chatbot built using a movie dataset from Kaggle. The bot provides personalized movie recommendations by leveraging a combination of **Sentence Transformers** for generating embeddings, **Qdrant** for vector storage and similarity search, and the **Llava-v1.5-7B** language model for generating dynamic responses.

## Features
- Personalized movie recommendations based on user preferences.
- Retrieval-based and generative approach combining **Sentence Embeddings** and **Language Models**.
- Efficient search using **Qdrant** for storing and querying movie data vectors.

## Architecture

1. **Language Model (LLM)**:  
   The system uses the `llava-v1.5-7b-q4.llamafile` model for generating responses.

2. **Encoder**:  
   Sentence embeddings are created using the `SentenceTransformer('all-MiniLM-L6-v2')` encoder, which is optimized for sentence-level encoding.

3. **Vector Database (Qdrant)**:  
   Qdrant is used for storing and querying vectors. The vectors are created by combining embeddings of the movie overview, genres, and keywords.

4. **Search and Filtering**:  
   - The model queries the vector database for relevant movie results based on cosine similarity.
   - Results are filtered with a threshold score of 0.5 to ensure relevance.


### Dataset
The movie dataset,[Movies Dataset on Kaggle](https://www.kaggle.com/datasets/utkarshx27/movies-dataset)  is sourced from Kaggle. 
