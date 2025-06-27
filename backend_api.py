from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
from flask_cors import CORS
from src.data_ingestion import process_uploaded_files
from src.retrieval_agent import get_response
from src.qa_memory import get_similar_answer, save_qa_pair
import os
from huggingface_hub import InferenceClient
from datetime import timedelta
import time
from flask_session import Session

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'your_strong_secret_key_here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['SESSION_TYPE'] = 'filesystem'  # Store sessions on the server filesystem
Session(app)
CORS(app)

MODELS = [
    {"id": "llama2_model", "name": "Llama2", "desc": "Llama2 70B"},
    {"id": "llama3_model", "name": "Llama3", "desc": "Llama3 8B"},
    {"id": "phi3_model", "name": "Phi-3", "desc": "Phi-3 Mini"},
    {"id": "deepseek_model", "name": "DeepSeek", "desc": "DeepSeek"}
]


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        # Replace this with real authentication logic
        if username and password:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])


@app.route('/models', methods=['GET'])
def models():
    return jsonify(MODELS)


@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json
    question = data['question']
    model_id = data.get('model_id')
    if not model_id:
        return jsonify({'error': 'No model selected. Please select a model.'}), 400

    # Greeting detection
    greetings = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'greetings', 'yo', 'hola', 'namaste', 'sup', 'hii', 'hiii', 'hiiii', 'helloo', 'hellooo'
    ]
    if any(word in question.lower().strip() for word in greetings):
        username = session.get('username', 'there')
        from datetime import datetime
        hour = datetime.now().hour
        if hour < 12:
            greet = 'Good morning'
        elif hour < 18:
            greet = 'Good afternoon'
        else:
            greet = 'Good evening'
        answer = f"{greet}, {username}! How can I assist you today?"
        from_memory = True
    else:
        model_mapping = {
            "llama2_model": "llama2",
            "llama3_model": "llama3",
            "phi3_model": "phi3",
            "deepseek_model": "Qwen/QwQ-32B"
        }
        ollama_model_name = model_mapping.get(model_id)
        similar_qa = get_similar_answer(question)
        if similar_qa:
            answer = similar_qa[1]
            from_memory = True
        else:
            answer = get_response(question, model_source=model_id, ollama_model_name=ollama_model_name)
            from_memory = False
    if 'chat_history' not in session:
        session['chat_history'] = []
    session['chat_history'].append({
        'question': question,
        'answer': answer,
        'from_memory': from_memory,
        'model_id': model_id,
        'timestamp': os.times().elapsed
    })
    session.modified = True
    return jsonify({'answer': answer, 'from_memory': from_memory})


@app.route('/chat_history', methods=['GET'])
def chat_history():
    if 'username' not in session:
        return jsonify([])
    return jsonify(session.get('chat_history', []))


@app.route('/upload', methods=['POST'])
def upload():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_filepath = os.path.join(temp_dir, file.filename)
    file.save(temp_filepath)

    # Track uploaded files in session
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []

    # Get file size and add to uploaded files list
    file_size_bytes = os.path.getsize(temp_filepath)
    if file_size_bytes < 1024:
        file_size = f"{file_size_bytes} B"
    elif file_size_bytes < 1024 * 1024:
        file_size = f"{round(file_size_bytes / 1024, 1)} KB"
    else:
        file_size = f"{round(file_size_bytes / (1024 * 1024), 1)} MB"

    session['uploaded_files'].append({
        'name': file.filename,
        'size': file_size,
        'path': temp_filepath,
        'upload_time': timedelta(seconds=int(os.times().elapsed)).total_seconds()
    })
    session.modified = True

    class UploadedFile:
        def __init__(self, path):
            self.name = os.path.basename(path)
            self._path = path

        def getbuffer(self):
            with open(self._path, "rb") as f:
                return f.read()

    uploaded_file = UploadedFile(temp_filepath)
    try:
        success = process_uploaded_files([uploaded_file])
        if success:
            return jsonify({
                'status': 'ok',
                'message': f'File "{file.filename}" uploaded and processed successfully!',
                'file_info': {
                    'name': file.filename,
                    'size': file_size
                }
            })
        return jsonify({'status': 'error', 'message': 'Processing failed'}), 500
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/uploaded_files', methods=['GET'])
def uploaded_files():
    """Return list of uploaded files for current user session"""
    if 'username' not in session:
        return jsonify([])
    return jsonify(session.get('uploaded_files', []))


@app.route('/clear_uploaded_files', methods=['POST'])
def clear_uploaded_files():
    """Clear all uploaded files for current user session"""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

    # Remove files from temp directory
    uploaded_files = session.get('uploaded_files', [])
    for file_info in uploaded_files:
        try:
            if os.path.exists(file_info.get('path', '')):
                os.remove(file_info['path'])
        except Exception as e:
            print(f"Error removing file {file_info.get('name', '')}: {e}")

    # Clear from session
    session['uploaded_files'] = []
    session.modified = True

    return jsonify({'status': 'ok', 'message': 'All uploaded files cleared'})


@app.route('/remove_file/<int:file_index>', methods=['DELETE'])
def remove_file(file_index):
    """Remove a specific uploaded file"""
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401

    uploaded_files = session.get('uploaded_files', [])
    if 0 <= file_index < len(uploaded_files):
        file_to_remove = uploaded_files[file_index]

        # Remove file from temp directory
        try:
            if os.path.exists(file_to_remove.get('path', '')):
                os.remove(file_to_remove['path'])
        except Exception as e:
            print(f"Error removing file {file_to_remove.get('name', '')}: {e}")

        # Remove from session
        del uploaded_files[file_index]
        session['uploaded_files'] = uploaded_files
        session.modified = True

        return jsonify({'status': 'ok', 'message': f'File "{file_to_remove.get("name", "")}" removed'})

    return jsonify({'status': 'error', 'message': 'File not found'}), 404


@app.route('/feedback', methods=['POST'])
def feedback():
    if 'username' not in session:
        return jsonify({'status': 'error', 'message': 'Not authenticated'}), 401
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    feedback = data.get('feedback')
    if question and answer and feedback:
        save_qa_pair(question, answer, feedback)
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'message': 'Missing data'}), 400


def stream_response(text, chunk_size=100, delay=0):
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        yield chunk


@app.route('/chat_stream', methods=['POST'])
def chat_stream():
    if 'username' not in session:
        return Response('Not authenticated', status=401)
    data = request.json
    question = data['question']
    model_id = data.get('model_id')
    if not model_id:
        return Response('No model selected. Please select a model.', status=400)
    model_mapping = {
        "llama2_model": "llama2",
        "llama3_model": "llama3",
        "phi3_model": "phi3",
        "deepseek_model": "Qwen/QwQ-32B"
    }
    ollama_model_name = model_mapping.get(model_id)
    similar_qa = get_similar_answer(question)
    if similar_qa:
        answer = similar_qa[1]
    else:
        answer = get_response(question, model_source=model_id, ollama_model_name=ollama_model_name)
    def generate():
        for chunk in stream_response(answer):
            yield chunk
    return Response(generate(), mimetype='text/plain')


if __name__ == '__main__':
    app.run(debug=True)
