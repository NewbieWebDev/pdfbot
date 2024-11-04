import os
import openai
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from flask import Flask, request, jsonify

app = Flask(__name__)

# Configuration
SEARCH_SERVICE_NAME = 'chatpdfsearch'  # Your Cognitive Search service name
INDEX_NAME = 'azureblob-indexer'         # Your index name
SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

# Initialize clients
search_client = SearchClient(
    endpoint=f"https://{SEARCH_SERVICE_NAME}.search.windows.net/",
    index_name=INDEX_NAME,
    credential=AzureKeyCredential(SEARCH_API_KEY)
)

openai.api_type = "azure"
openai.api_key = OPENAI_API_KEY
openai.api_base = OPENAI_ENDPOINT
openai.api_version = "2023-03-15-preview"

def search_pdfs(query):
    results = search_client.search(query, top=5)
    contents = []
    for result in results:
        contents.append(result['content'])
    return "\n".join(contents)

def generate_response(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = openai.Completion.create(
        engine="davinci",  # Use the appropriate engine name
        prompt=prompt,
        max_tokens=150,
        temperature=0.2,
        stop=["\n"]
    )
    return response.choices[0].text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    # Search relevant PDF content
    context = search_pdfs(question)
    
    # Generate response using OpenAI
    answer = generate_response(context, question)
    
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
