# 🫀 Crag-chatbot-for-Cardiology-Related-Queries:
Cardiology-focused Corrective-RAG chatbot with document retrieval and web fallback using Tavily, TinyLLaMA and Streamlit.


CRAG (Corrective Retrieval-Augmented Generation) is a domain-specific AI chatbot designed to assist users with questions related to heart health, treatment, diagnosis, and lifestyle. It leverages a hybrid approach—retrieving answers from medical literature and falling back on real-time web search when necessary.


## 🚀 Live Demo:  
🔗 *[Try on Hugging Face Spaces](https://huggingface.co/spaces/88rehaan88/crag-chatbot)*  

## 💡 Features:
🔎 Domain-Specific Retrieval: Uses heart-health-related documents for accurate and trusted responses.

📚 TinyLlama Generator: Generates clear, bullet-point medical responses with TinyLlama 1.1B.

🧠 Relevance Filtering: Grades document relevance using FLAN-T5 before generating answers.

🌐 Web Search Fallback: Falls back on the Tavily web search API when local docs are not relevant.

🧑‍⚕️ Medical UX: Streamlit-based interface styled for a clean, informative user experience.
