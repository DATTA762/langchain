
# from fastapi import FastAPI
# from pydantic import BaseModel
# from groq import Groq

# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_huggingface import HuggingFaceEmbeddings

# from sentence_transformers import CrossEncoder
# import faiss
# import os


# # 2. FastAPI App

# app = FastAPI(title="RAG API with Groq")


# # 3. Request Schema
# class QueryRequest(BaseModel):
#     query: str


# # 4. Load Models (ONCE)

# print(" Loading models...")

# # Embeddings
# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     encode_kwargs={"normalize_embeddings": True}
# )

# # Cross-Encoder
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# # Groq Client
# client = Groq(api_key="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")


# # 5. Load & Index Documents
# FILE_PATH = "usa2.txt"

# loader = TextLoader(FILE_PATH, encoding="utf-8")
# docs = loader.load()

# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# split_docs = splitter.split_documents(docs)

# embedding_dim = 768
# index = faiss.IndexFlatIP(embedding_dim)

# vector_store = FAISS(
#     embedding_function=embedding_model,
#     index=index,
#     docstore=InMemoryDocstore(),
#     index_to_docstore_id={}
# )

# vector_store.add_documents(split_docs)

# retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# print(" RAG system ready!")


# # 6. Re-ranking Function
# def rerank(query, docs, top_k=3):
#     pairs = [(query, doc.page_content) for doc in docs]
#     scores = reranker.predict(pairs)

#     ranked = sorted(
#         zip(scores, docs),
#         key=lambda x: x[0],
#         reverse=True
#     )

#     # threshold filtering
#     filtered = [r for r in ranked if r[0] > 0.3]

#     if len(filtered) == 0:
#         filtered = ranked[:2]

#     return filtered[:top_k]


# # 7. Groq LLM Function
# def generate_answer(query, context):
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are a factual assistant. Answer only from context."
#             },
#             {
#                 "role": "user",
#                 "content": f"""
# Context:
# {context}

# Question:
# {query}

# Answer in 1-2 lines. If not found, say 'Not found in the document'
# """
#             }
#         ]
#     )

#     return response.choices[0].message.content


# # 8. API Endpoint
# @app.post("/ask")
# def ask_question(request: QueryRequest):

#     query = request.query

#     # Step 1: Retrieve
#     initial_docs = retriever.invoke(query)

#     # Step 2: Re-rank
#     reranked = rerank(query, initial_docs)
#     final_docs = [doc for _, doc in reranked]

#     # Step 3: Build Context
#     context_text = "\n\n".join([doc.page_content for doc in final_docs])

#     # Step 4: Generate Answer
#     answer = generate_answer(query, context_text)

#     return {
#         "query": query,
#         "answer": answer,
#         "sources": [doc.metadata for doc in final_docs]
#     }
# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import faiss
import os
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set in environment")

# 2. FastAPI App
app = FastAPI(title="RAG API with Groq")

# 3. Request Schema
class QueryRequest(BaseModel):
    query: str

# 4. Load Models (ONCE)
print("Loading models...")

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Cross-Encoder reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# 5. Load & Index Documents
FILE_PATH = "usa2.txt"  # make sure this file is in your Render project

loader = TextLoader(FILE_PATH, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

embedding_dim = 768
index = faiss.IndexFlatIP(embedding_dim)

vector_store = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(split_docs)
retriever = vector_store.as_retriever(search_kwargs={"k": 10})

print("RAG system ready!")

# 6. Reranking Function
def rerank(query, docs, top_k=3):
    if not docs:
        return []
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    filtered = [doc for score, doc in ranked if score > 0.3]
    return filtered[:top_k] if filtered else [doc for _, doc in ranked[:2]]

# 7. Groq LLM Function
def generate_answer(query, context):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a factual assistant. Answer only from context."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query}

Answer in 1-2 lines. If not found, say 'Not found in the document'
"""
            }
        ]
    )
    # Some SDKs return .choices[0].message.content, others .choices[0].text
    try:
        return response.choices[0].message.content
    except AttributeError:
        return response.choices[0].text

# 8. API Endpoint
@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Step 1: Retrieve
    initial_docs = retriever.get_relevant_documents(query)

    # Step 2: Re-rank
    final_docs = rerank(query, initial_docs)

    # Step 3: Build Context
    context_text = "\n\n".join([doc.page_content for doc in final_docs])

    # Step 4: Generate Answer
    answer = generate_answer(query, context_text)

    return {
        "query": query,
        "answer": answer,
        "sources": [doc.metadata for doc in final_docs]
    }
