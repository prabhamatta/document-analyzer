from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ai import generateText
from ai_sdk.openai import openai

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# In-memory database to store document embeddings
document_embeddings = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    _, extension = os.path.splitext(file_path)
    if extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif extension == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return ' '.join([page.extract_text() for page in reader.pages])
    elif extension == '.docx':
        doc = Document(file_path)
        return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return ' '.join([p.text for p in soup.find_all('p')])

def process_document(document_id, text):
    # Split the text into chunks (you may need to implement a more sophisticated chunking method)
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    
    # Generate embeddings for each chunk
    embeddings = model.encode(chunks)
    
    # Store the embeddings
    document_embeddings[document_id] = {
        'chunks': chunks,
        'embeddings': embeddings
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_text_from_file(file_path)
        process_document(filename, text)
        return jsonify({'message': 'File uploaded and processed successfully'}), 200
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/process_url', methods=['POST'])
def process_url():
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'No URL provided'}), 400
    try:
        text = extract_text_from_url(url)
        process_document(url, text)
        return jsonify({'message': 'URL processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query])[0]
    
    all_similarities = []
    for doc_id, doc_data in document_embeddings.items():
        similarities = cosine_similarity([query_embedding], doc_data['embeddings'])[0]
        top_indices = np.argsort(similarities)[-top_k:]
        for idx in top_indices:
            all_similarities.append((doc_id, idx, similarities[idx]))
    
    all_similarities.sort(key=lambda x: x[2], reverse=True)
    
    relevant_chunks = []
    for doc_id, chunk_idx, _ in all_similarities[:top_k]:
        relevant_chunks.append(document_embeddings[doc_id]['chunks'][chunk_idx])
    
    return relevant_chunks

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    relevant_chunks = get_relevant_chunks(query)
    context = ' '.join(relevant_chunks)
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = generateText({
            "model": openai("gpt-4o"),
            "prompt": prompt,
            "max_tokens": 150  # Adjust as needed
        })
        
        return jsonify({'answer': response['text']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)