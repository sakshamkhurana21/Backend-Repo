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

directory = "doc"


app = Flask(__name__)


@app.route('/api/extract-information', methods=['POST'])
def extract_information():
    query = request.json.get('query')
    emb = get_similar_docs(query)
    summary= summarize_text(emb)
    final = summary_to_chatgpt(summary, query)
    return {'ChatGPt Response': final }


            
def get_similar_docs(query):
    print(31)
    if query:
        docs = []
        print(os.listdir())
        for file in os.listdir(directory):
            if file[0] != ".":
                loader = TextLoader(directory + '/' + file)
                print("loader: ",loader)
                docs.extend(loader.load())
        embeddings = HuggingFaceEmbeddings()
        db = FAISS.from_documents(docs, embeddings)
        docs = db.similarity_search(query)
        return docs[0:len(docs)//2]
    return ""



def summarize_text(text_list):
    array=[]
    for items in text_list:
        array.append(items.page_content)
    text = "-----END OF DOCUMENT-----".join(array)
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": ""
    }
    data = {
        "messages" : [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f'Summarise the following documents. Ensure you summarise one document at a time and retain the structure. The documents to summarise are: {text}'}],
        "temperature" : 0.3,
        "max_tokens": 100,
        "model": "gpt-3.5-turbo"
    }
    response = requests.post(endpoint, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def summary_to_chatgpt(text, query):
    api_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': ''
    }
    data = {
        "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": f'{text}\nQuery: {query}. Can you reply to the query based on the summary provided.'}],
        'temperature': 0.7,
        'max_tokens': 100,
        "model": "gpt-3.5-turbo"
    }
    response = requests.post(api_url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


if __name__ == '__main__':
    app.run()
