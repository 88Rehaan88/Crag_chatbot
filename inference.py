# Importing the necessary libraries:
import os
import shutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient

# Loading the documents:
def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
                if content:
                    documents.append(Document(page_content=content, metadata={"source": filename}))
    return documents

# Splitting the documents into chunks: 
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    return splitter.split_documents(documents)

# Loading the embedding model and initialize the vectorstore for retrieval:
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Load and embed:
if not os.path.exists("chroma_db"):
    docs = load_documents("Rag_Documents")  # Load heart-related docs for embedding
    chunks = split_documents(docs)
    chunks = split_documents(docs)
    shutil.rmtree("chroma_db", ignore_errors=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db",
        collection_name="heart-docs"
    )
else:
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model,
        collection_name="heart-docs"
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Setting up the retriever to return the top 2 most relevant chunks

# Use FLAN-T5 to check if a retrieved chunk is relevant to the query or not:
from transformers import AutoModelForSeq2SeqLM
flan_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
flan_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cuda" if torch.cuda.is_available() else "cpu")

def grade_relevance(query, passage):
    prompt = f"""Is the following passage relevant to the question?\n
Question: {query}\n
Passage: {passage}\n
Answer yes or no."""
    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(flan_model.device)
    output = flan_model.generate(**inputs, max_new_tokens=2)
    result = flan_tokenizer.decode(output[0], skip_special_tokens=True)
    return result.strip().lower() == "yes"

# Loading TinyLLaMA model for generation: 
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
llm_tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Generate an answer using TinyLLaMA based on retrieved context and query:
def generate_with_model(context_docs, query):
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = f"""
You are a helpful medical assistant. Based on the provided context, answer the user's question as clearly and concisely as possible.
Use bullet points if listing recommendations or symptoms. Do not repeat the question. Avoid any irrelevant content like website menus or links.

Context:
{context}

Question: {query}
Answer:"""
    inputs = llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(llm_model.device)
    output = llm_model.generate(**inputs, max_new_tokens=300)
    full_response = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full_response.split("Answer:")[-1].strip()
    return answer

# Fallback to Tavily web search if no relevant local docs are found:
hf_token = os.getenv("HF_TOKEN")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Search the web using Tavily API:
def search_web(query):
    response = tavily.search(query=query, search_depth="basic", max_results=3)
    top_snippets = [r["content"] for r in response["results"]]
    return "\n\n".join(top_snippets)

# Main pipeline: grade, retrieve, generate or fallback to web and generate:
def ask_question(query: str) -> str:
    retrieved_docs = retriever.get_relevant_documents(query)
    filtered_docs = [doc for doc in retrieved_docs if grade_relevance(query, doc.page_content)]

    print(f"Retrieved: {len(retrieved_docs)}, Relevant: {len(filtered_docs)}")

    if filtered_docs:
        print("✅ Using local documents for response...")
        answer = generate_with_model(filtered_docs, query)
    else:
        print("🌐 No relevant documents found. Searching the web...")
        web_context = search_web(query)
        fake_doc = type("Doc", (), {"page_content": web_context})()
        answer = generate_with_model([fake_doc], query)

    print("\n💬 Final Answer:\n", answer)
    return answer
