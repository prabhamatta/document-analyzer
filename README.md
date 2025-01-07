# document-analyzer
 Web application where you can input text in multiple forms. And use chatgpt api to answer questions related to the text.

1. Handles file uploads and URL processing
2. Extracts text from various document types (TXT, PDF, DOCX) and websites
3. Implements a RAG process using sentence embeddings for efficient retrieval
4. Provides a chat endpoint that uses the RAG process to generate relevant answers
5. Includes a simple chat widget that can be integrated into your website


To use this application:

1. Install the required dependencies:

```plaintext
pip install flask werkzeug requests beautifulsoup4 PyPDF2 python-docx sentence-transformers scikit-learn numpy ai-sdk
```


2. Set up your OpenAI API key as an environment variable:

```plaintext
export OPENAI_API_KEY=your_api_key_here
```


3. Run the Flask application:

```plaintext
python app.py
```


4. Access the chat widget at `http://localhost:5000`
5. To upload a document or process a URL, you can create additional routes or use tools like cURL or Postman to send POST requests to the `/upload` and `/process_url` endpoints.