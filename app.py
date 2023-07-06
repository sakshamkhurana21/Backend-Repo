# from transformers import AutoTokenizer
import requests
from flask import Flask, request, jsonify
# from retry import retry
import numpy as np 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader
import os
from langchain.embeddings import HuggingFaceEmbeddings

directory = "doc/"


app = Flask(__name__)

# model_id = "sentence-transformers/all-MiniLM-L6-v2"
# api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
# headers = {"Authorization": f"Bearer {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}"}


# # ChatGPT API endpoint
# CHATGPT_API_URL = 'https://api.openai.com/v1/engines/davinci-codex/completions'
# CHATGPT_API_KEY = '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'  


# @app.route('/api/extract-information', methods=['POST'])
# def extract_information():
#     query = request.json.get('query')
#     if query:
#         query_embedding = preprocess_and_convert_to_vector(query)  # Implement this function
#         results = retrieve_documents_with_similarity(query_embedding)
#         sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
#         k = 6
#         summarized_results = summarize_top_k(sorted_results[:k])
#         return jsonify({'results': summarized_results})

#     return jsonify({'results': []})

@app.route('/api/extract-information', methods=['POST'])
def extract_information():
    query = request.json.get('query')
    if query:
        docs = []
        for file in os.listdir(directory):
            if file != ".ipynb_checkpoints":
                loader = TextLoader(directory + '/' + file)
        docs.extend(loader.load())
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        docs = db.similarity_search(query)
        return docs[0].page_content
    return []

            

# tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# def preprocess_and_convert_to_vector(query):
#     tokenized_input = tokenizer.encode_plus(query, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
#     input_ids = tokenized_input['input_ids']
#     attention_mask = tokenized_input['attention_mask']
#     return input_ids, attention_mask



# def retrieve_documents_with_similarity(query_input_ids, query_attention_mask):
#     query_input_ids = query_input_ids.numpy().astype('float32')
#     query_attention_mask = query_attention_mask.numpy().astype('float32')
#     query_vector = np.concatenate((query_input_ids, query_attention_mask), axis=1)
#     k = 10  # Number of nearest neighbors to retrieve
#     distances, indices = index.search(query_vector, k)

#     results = []
#     for i, index in enumerate(indices[0]):
#         score = 1 / (1 + distances[0][i])  # Calculating the similarity score from the distance
#         document = retrieve_document_by_index(index)  
#         results.append({'document': document, 'score': score})
#     return results

# def retrieve_document_by_index(index):



# def summarize_top_k(results):
#     summarized_results = []
#     for result in results:
#         text = tokenizer.decode(result['input_ids'])
#         summary = send_to_chatgpt(text)
#         summarized_results.append(summary)
#     return summarized_results

# def send_to_chatgpt(text):
#     headers = {
#         'Content-Type': 'application/json',
#         'Authorization': f'Bearer {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}'
#     }
#     payload = {
#         'prompt': text,
#         'max_tokens': 50  
#     }
#     response = requests.post(CHATGPT_API_URL, headers=headers, json=payload)
#     return response.json()['choices'][0]['text']

if __name__ == '__main__':
    app.run()
