from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables import RunnableLambda
from .utils.config_loader import get_config
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv
import requests
from huggingface_hub import InferenceClient
import pandas as pd
from pathlib import Path
from langchain.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredXMLLoader, TextLoader

load_dotenv()

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HUGGINGFACE_API_KEY = "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"

# --- Module-level initialization ---
model_config = get_config("model_config")
embedding_model_info = model_config.get("embedding_model", {})
model_name = embedding_model_info.get("repo_id", "sentence-transformers/all-MiniLM-L6-v2")
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 1})

def get_retriever():
    return retriever

def reload_retriever():
    global db, retriever
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 1})

def format_context_with_references(chunks):
    formatted = []
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        page = chunk.metadata.get('page', None)
        ref = f"[{source}{', page ' + str(page) if page else ''}]"
        formatted.append(f"{ref} {chunk.page_content}")
    return '\n'.join(formatted)

def get_ollama_llm(model_name="llama2", temperature=0.7, max_tokens=500):
    return OllamaLLM(model=model_name, temperature=temperature, max_tokens=max_tokens)

def query_huggingface_inference_api(prompt, model, max_tokens=500, temperature=0.7):
    HUGGINGFACE_API_KEY = "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"
    HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature
        }
    }
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    if isinstance(result, list) and "generated_text" in result[0]:
        return result[0]["generated_text"]
    elif "generated_text" in result:
        return result["generated_text"]
    elif "text" in result:
        return result["text"]
    else:
        return str(result)

def query_fireworks_deepseek(prompt, model, max_tokens=3000, temperature=0.7):
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY is not set in the environment. Please set it in your .env file or environment variables.")
    client = InferenceClient(
        provider="fireworks-ai",
        api_key=api_key,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return completion.choices[0].message.content

def create_retrieval_chain(model_source=None, ollama_model_name=None):
    retriever = get_retriever()
    if not retriever:
        return None
    prompts_config = get_config("prompt_templates")
    response_prompt_template = prompts_config.get("response_prompt")
    prompt = ChatPromptTemplate.from_template(response_prompt_template)
    model_config = get_config("model_config")
    if not model_source or model_source not in model_config:
        raise ValueError(f"Unknown model_source '{model_source}'. Please select a valid model from the UI.")
    model_info = model_config[model_source]
    provider = model_info.get("provider")
    model = model_info.get("model")
    temperature = model_info.get("temperature", 0.7)
    max_length = model_info.get("max_tokens", 1000)

    if provider == "ollama":
        llm = get_ollama_llm(model_name=model, temperature=temperature, max_tokens=max_length)
        def llm_call_fn(inputs):
            return llm(inputs["question"] + "\n" + inputs["context"])
    elif provider == "huggingface":
        def llm_call_fn(inputs):
            full_prompt = prompt.format(question=inputs["question"], context=inputs["context"])
            return query_huggingface_inference_api(full_prompt, model=model, max_tokens=max_length, temperature=temperature)
    elif provider == "fireworks":
        def llm_call_fn(inputs):
            full_prompt = prompt.format(question=inputs["question"], context=inputs["context"])
            return query_fireworks_deepseek(full_prompt, model=model, max_tokens=max_length, temperature=temperature)
    elif provider == "featherless-ai":
        def llm_call_fn(inputs):
            full_prompt = prompt.format(question=inputs["question"], context=inputs["context"])
            return query_huggingface_inference_api(full_prompt, model=model, max_tokens=max_length, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    def format_input(inputs):
        if isinstance(inputs["context"], list):
            inputs["context"] = format_context_with_references(inputs["context"])
        return inputs
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(format_input)
        | RunnableLambda(llm_call_fn)
    )
    return chain

def get_response(query: str, model_source="Hugging Face Hub", ollama_model_name=None):
    chain = create_retrieval_chain(model_source=model_source, ollama_model_name=ollama_model_name)
    if not chain:
        return "The document database has not been created yet. Please upload a PDF first."
    return chain.invoke(query)

def load_documents(temp_filepath):
    ext = Path(temp_filepath).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(temp_filepath)
    elif ext == ".docx" or ext == ".doc":
        loader = UnstructuredWordDocumentLoader(temp_filepath)
    elif ext == ".xml":
        loader = UnstructuredXMLLoader(temp_filepath)
    elif ext == ".txt":
        loader = TextLoader(temp_filepath)
    elif ext == ".xlsx":
        df = pd.read_excel(temp_filepath)
        text = df.to_string()
        return [{"page_content": text, "metadata": {}}]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()

if __name__ == '__main__':
    # Example usage:
    test_query = "What is this document about?"
    print(f"Query: {test_query}")
    response = get_response(test_query)
    print(f"Response: {response}") 