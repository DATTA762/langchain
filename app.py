from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Fast RAG API")

class QueryRequest(BaseModel):
    query: str

print("Loading models...")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 10})

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = Groq(api_key=GROQ_API_KEY)

print("RAG ready!")

def rerank(query, docs):
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:3]]

def generate_answer(query, context):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":"Answer only from context"},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion:{query}"}
        ]
    )

    return response.choices[0].message.content

@app.post("/ask")
def ask(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    docs = retriever.invoke(query)

    reranked = rerank(query, docs)

    context = "\n\n".join([d.page_content for d in reranked])

    answer = generate_answer(query, context)

    return {
        "query": query,
        "answer": answer
    }
