import sqlite3
import os
from typing import Optional, Tuple, List

try:
    from sentence_transformers import SentenceTransformer, util
    import numpy as np
    _HAS_EMBEDDINGS = True
    _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    _HAS_EMBEDDINGS = False
    _MODEL = None

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'qa_memory.db')


def init_db():
    """Initialize the SQLite database and create the qa_memory table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS qa_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    feedback TEXT,
                    embedding BLOB
                )''')
    conn.commit()
    conn.close()


def save_qa_pair(question: str, answer: str, feedback: str = 'yes'):
    """Save a Q&A pair with feedback and optional embedding."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    embedding = None
    if _HAS_EMBEDDINGS:
        embedding = _MODEL.encode(question)
        embedding = embedding.tobytes()
    c.execute('INSERT INTO qa_memory (question, answer, feedback, embedding) VALUES (?, ?, ?, ?)',
              (question, answer, feedback, embedding))
    conn.commit()
    conn.close()


def get_similar_answer(query: str, threshold: float = 0.8) -> Optional[Tuple[str, str]]:
    """
    Retrieve the most similar question from the DB and its answer.
    Returns (question, answer) if similarity above threshold, else None.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, question, answer, embedding FROM qa_memory WHERE feedback = "yes"')
    rows = c.fetchall()
    conn.close()
    if not rows:
        return None
    if _HAS_EMBEDDINGS:
        query_emb = _MODEL.encode(query)
        best_score = -1
        best_row = None
        for row in rows:
            db_emb = row[3]
            if db_emb is not None:
                try:
                    db_emb_np = np.frombuffer(db_emb, dtype=np.float32)
                    # Ensure shapes match
                    if db_emb_np.shape == query_emb.shape:
                        score = util.cos_sim(query_emb, db_emb_np).item()
                    else:
                        continue
                except Exception:
                    continue
                if score > best_score:
                    best_score = score
                    best_row = row
        if best_score >= threshold:
            return best_row[1], best_row[2]
    # Fallback: simple string similarity
    from difflib import SequenceMatcher
    def sim(a, b):
        return SequenceMatcher(None, a, b).ratio()
    best_score = -1
    best_row = None
    for row in rows:
        score = sim(query, row[1])
        if score > best_score:
            best_score = score
            best_row = row
    if best_score >= threshold:
        return best_row[1], best_row[2]
    return None


def update_feedback(question: str, feedback: str):
    """Update feedback for a given question (exact match)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE qa_memory SET feedback = ? WHERE question = ?', (feedback, question))
    conn.commit()
    conn.close()

# Call this once at app startup
init_db() 