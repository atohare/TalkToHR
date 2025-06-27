import streamlit as st
from src.data_ingestion import process_uploaded_files
from src.retrieval_agent import get_response
import os
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.qa_memory import get_similar_answer, save_qa_pair

load_dotenv()

st.title("AI PDF Chatbot")

# Sidebar for model selection and file upload
with st.sidebar:
    st.header("Settings")
    model_source = st.selectbox(
        "Choose LLM Source",
        options=["Hugging Face Hub", "Ollama (local server)", "DeepSeek (Fireworks)", "Google Flan-T5 (Hugging Face)"],
        index=0
    )
    ollama_model_name = None
    if model_source == "Ollama (local server)":
        ollama_models = ["mistral", "llama2"]
        selected_ollama_model = st.selectbox("Select Ollama Model", ollama_models, index=0)
        # custom_ollama_model = st.text_input("Or enter custom Ollama Model Name", value=selected_ollama_model)
        # ollama_model_name = custom_ollama_model
        # Map to correct model_source key for backend
        if ollama_model_name and ollama_model_name.lower() == "llama2":
            backend_model_source = "llama2_model"
        elif ollama_model_name and ollama_model_name.lower() == "mistral":
            backend_model_source = "mistral_model"
        else:
            backend_model_source = "llama2_model"  # fallback to llama2_model if unknown
    elif model_source == "DeepSeek (Fireworks)":
        backend_model_source = "deepseek_model"
    elif model_source == "Google Flan-T5 (Hugging Face)":
        backend_model_source = "google_flan_t5_model"
    else:
        backend_model_source = "llama2_model"  # fallback to a valid model if needed
    st.header("Upload PDF")
    uploaded_files = st.file_uploader(
        "Upload your PDFs, DOCX, XML, TXT, or DOC files and click 'Process'",
        accept_multiple_files=True,
        type=["pdf", "docx", "xml", "txt", "doc"]
    )
    if st.button("Process"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                if process_uploaded_files(uploaded_files):
                    st.success("Documents processed successfully!")
                else:
                    st.error("Failed to process documents.")
        else:
            st.warning("Please upload at least one PDF file.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # 1. Check for similar Q&A in memory
    similar_qa = get_similar_answer(prompt)
    use_saved_answer = False
    if similar_qa:
        similar_q, similar_a = similar_qa
        with st.chat_message("assistant"):
            st.markdown(f"**Found a similar question in memory:**\n> {similar_q}\n\n**Saved answer:**\n{similar_a}")
            use_saved_answer = st.radio(
                "Is this answer sufficient?",
                ("Yes, this is good", "No, give me a new answer"),
                key=f"feedback-similar-{len(st.session_state.messages)}"
            ) == "Yes, this is good"
        if use_saved_answer:
            st.session_state.messages.append({"role": "assistant", "content": similar_a})
    if not similar_qa or not use_saved_answer:
        # 2. Get LLM response as usual
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_response(prompt, model_source=backend_model_source, ollama_model_name=ollama_model_name)
                st.markdown(response)
                # 3. Ask for feedback
                feedback = st.radio(
                    "Are you satisfied with this answer?",
                    ("Yes", "No"),
                    key=f"feedback-llm-{len(st.session_state.messages)}"
                )
                if feedback == "Yes":
                    save_qa_pair(prompt, response, feedback="yes")
        st.session_state.messages.append({"role": "assistant", "content": response})

def get_llamacpp_llm(model_path, n_ctx=2048, n_threads=4):
    return LlamaCpp(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        temperature=0.7,
        max_tokens=1000,
        verbose=True
    ) 

def create_retrieval_chain(model_source="Hugging Face Hub", ollama_model_name=None):
    # ... retriever code ...
    if model_source == "Ollama (local server)":
        llm = get_ollama_llm(model_name=ollama_model_name or "llama2")
    else:
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": temperature, "max_length": max_length},
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
    # ... rest of chain ... 