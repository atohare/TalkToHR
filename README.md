# Generative AI PDF Chatbot

This project is a PDF chatbot that allows you to upload documents and ask questions about their content.  
It's built with **Python, Flask, LangChain, ChromaDB, HuggingFace, and Ollama**.

---

## Tech Stack

- **Backend:** Python, Flask, Flask-CORS
- **Frontend:** Flask (Jinja2 templates), HTML, CSS, JavaScript
- **Vector Store:** ChromaDB
- **Embeddings:** HuggingFace Sentence Transformers
- **LLMs Supported:** Ollama  offline models(Llama2, Llama3, Phi-3), HuggingFace api model (Deepseek)
- **Prompt Engineering:** Custom YAML templates
- **Feedback/RLHF:** SQLite-based Q&A memory with user feedback
- **Logging:** (Add your logging config if you want more details)
- **Chat History:** Session-based, per user

---

## Features & Functionality

### 1. **User Authentication**
- Simple login system (username only, can be extended).

### 2. **Document Upload & Ingestion**
- Upload PDFs, DOCX, XML, TXT, or DOC files.
- Automatic chunking and embedding of documents.
- Vector store (ChromaDB) is created and persisted for fast retrieval.

### 3. **Chat with Retrieval-Augmented Generation**
- Ask questions about your uploaded documents.
- The system retrieves the most relevant chunks and passes them to the LLM for context-aware answers.

### 4. **Model Selection**
- Choose between Llama2, Llama3, Phi-3 (Ollama), DeepSeek, .
- All models are configured for a max response length of 500 tokens.

### 5. **Chat History**
- All user questions and bot answers are stored in session chat history.
- Chat history is available per user session.

### 6. **Feedback & RLHF (Reinforcement Learning from Human Feedback)**
- After each answer, users can provide feedback (👍/👎).
- Feedback is stored in a local SQLite database.
- The system uses this feedback to improve future answers (retrieves similar Q&A pairs with positive feedback).

### 7. **Logging**
- (Add details if you have a logging_config.yaml or use Python logging for errors, uploads, etc.)

### 8. **Prompt Engineering**
- Prompts for both routing and answering are configurable in `config/prompt_templates.yaml`.

### 9. **File Management**
- Upload, list, and remove files.
- Uploaded files are tracked per user session.

---

## Project Structure

```
generative_ai_project/
├          
├── backend_api.py          # Main Flask backend and UI
├── config/                 # Configuration files
│   ├── logging_config.yaml # logging 
│   ├── model_config.yaml   # models config
│   └── prompt_templates.yaml  # Prompts template 
├── data/                   # Data storage (e.g., for ChromaDB)
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_ingestion.py   # Handles PDF processing and embedding
│   ├── retrieval_agent.py  # Core retrieval and generation logic
│   ├── qa_memory.py         # Q&A memory, feedback, RLHF
│   ├── handlers/
│   ├── llm/
│   ├── prompt_engineering/
│   └── utils/
├── static/                 # Static files (JS, CSS, avatars)
│   ├── chatbot.js
│   ├── chatbot.css
│   ├── bot.png
│   └── user.png
├── templates/              # Flask HTML templates
│   ├── index.html
│   └── login.html
├── requirements.txt
└── README.md
```

---

## How it Works (Step-by-Step)

1. **Login:** Enter your username to start a session.
2. **Upload Documents:** Upload PDFs, DOCX, XML, TXT, or DOC files.
3. **Embedding & Vector Store:** Documents are split, embedded, and stored in ChromaDB.
4. **Ask Questions:** Type your question in the chat UI.
5. **Retrieval:** The system finds the most relevant document chunks.
6. **LLM Answering:** The selected LLM (Llama2, Llama3, Phi-3, etc.) generates an answer using the retrieved context.
7. **Feedback:** After each answer, provide feedback (👍/👎) to help improve future responses.
8. **Chat History:** View your previous questions and answers in the session.

---

## Running the Application

1. **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd generative_ai_project
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your environment variables:**
    Create a `.env` file and add your Hugging Face API key if needed:
    ```
    HUGGINGFACEHUB_API_TOKEN="your_huggingface_api_token_here"
    ```

5. **Start the Flask backend and UI:**
    ```bash
    python backend_api.py
    ```

6. **Open your browser:**
    Go to [http://localhost:5000](http://localhost:5000)

---

## Best Functionalities

- **RLHF/Feedback:** User feedback is stored and used to improve answer quality.
- **Chat History:** All Q&A pairs are stored per session and can be retrieved.
- **Retrieval-Augmented Generation:** Combines vector search with LLMs for accurate, context-aware answers.
- **Multi-model Support:** Easily switch between Llama2, Llama3, Phi-3, and more.
- **Prompt Engineering:** Easily customize prompts for routing and answering.
- **Logging:** (Add details if you have logging enabled.)

---

## Supported Models

- **Llama2** (Ollama)
- **Llama3** (Ollama)
- **Phi-3** (Ollama)
- **DeepSeek** (Fireworks)


---

## Notes
- The UI is now Flask-based (not Streamlit).
- All dependencies are listed in `requirements.txt`.
- For local LLMs, make sure you have Ollama and the desired models (e.g., phi3, llama2, llama3) installed.

## Using Ollma models :

ollama pull llama2

# For Llama 3
ollama pull llama3

# For Phi-3
ollama pull phi3




## Using Llama.cpp Quantized Models

To use the Llama.cpp (local quantized) option, you need to download a `.gguf` model file:

1. Go to [TheBloke/Llama-2-7B-Chat-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF).
2. Download a file like `llama-2-7b-chat.Q4_K_M.gguf`.
3. Place the downloaded `.gguf` file in the `models/` directory in your project root (create the directory if it doesn't exist).

Example (using command line):
```bash
mkdir -p models
wget -O models/llama-2-7b-chat.Q4_K_M.gguf https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
```

**Note:** The path to the model is static and set to `models/llama-2-7b-chat.Q4_K_M.gguf` in the code. You do not need to enter it in the UI.
