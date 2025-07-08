# 🫀 Crag-chatbot-for-Cardiology-Related-Queries:


CRAG (Corrective Retrieval-Augmented Generation) is a domain-specific AI chatbot designed to assist users with questions related to heart health, treatment, diagnosis, and lifestyle. It leverages a hybrid approach—retrieving answers from medical literature and falling back on real-time web search when necessary.


## 🚀 Live Demo:  
🔗 *[Try on Hugging Face Spaces](https://huggingface.co/spaces/88rehaan88/crag-chatbot)*  

## 🖼️ Demo Screenshots:

- Local Documents Response:
<img src="https://github.com/user-attachments/assets/e826d5ca-0f9d-4cbe-8840-4d9819757cdf" width="800">

- <img src="https://github.com/user-attachments/assets/f0f42371-a491-4814-9669-9474ee13fbac" width="800">



- Web Fallback Triggered: 
<img src="https://github.com/user-attachments/assets/670ffef0-bb8c-4c9b-b452-8574e508f094" width="800">


<img src="https://github.com/user-attachments/assets/d7fadeab-f3ff-4186-865b-ab4a0881ba16" width="800">
## 💡 Features:
🔎 Domain-Specific Retrieval: Uses heart-health-related documents for accurate and trusted responses.

📚 TinyLlama Generator: Generates clear, bullet-point medical responses with TinyLlama 1.1B.

🧠 Relevance Filtering: Grades document relevance using FLAN-T5 before generating answers.

🌐 Web Search Fallback: Falls back on the Tavily web search API when local docs are not relevant.

🧑‍⚕️ Medical UX: Streamlit-based interface styled for a clean, informative user experience.


## 🛠️ Tech Stack:
- RAG Framework: LangChain + ChromaDB

- Embedding Model: BAAI/bge-small-en

- Relevance Grader: google/flan-t5-base

- LLM Generator: TinyLlama/TinyLlama-1.1B-Chat-v1.0

- Web Search: Tavily API

- Frontend: Streamlit

## ✅ Highlights:
- Tiny but Mighty: Optimized with TinyLlama for minimal resource use.

- Fail-Safe: Works even when local docs are irrelevant.

## 🔮 Potential Improvements:
- Model Upgrade: Replace TinyLlama with a stronger model like Mistral-7B or Zephyr for more nuanced medical generation.
  
- Document Uploader: Let users upload their own PDFs or clinical notes for personalized RAG responses.

- Answer Evaluation: Integrate QA evaluation metrics such as BLEU, ROUGE, or semantic similarity scoring (e.g., BERTScore) to automatically assess response quality.

# 🧾 Conclusion:
The CRAG Chatbot is a fully functional, lightweight, domain-specific assistant built with real-world architecture. It combines efficient retrieval with a fallback mechanism for general questions, demonstrating how small models can be production-ready when backed by good design. This project can also be scaled into a larger multi-agent systems. 
