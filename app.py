from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from src.prompt import *
import os

app = Flask(__name__)
load_dotenv()
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

embeddings=download_hugging_face_embeddings()

index_name="medchatbot"

docsearch=PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever =docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

llm= Ollama(
    model="mistral"
)
prompt= ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')
chat_history=[]
@app.route("/", methods=["GET","POST"])
def chat():
    global chat_history
    if request.method == "POST":
        user_input = request.form["user_input"]
        chat_history.append({"sender": "user", "text": user_input})

        try:
            response = rag_chain.invoke({"input": user_input})
            bot_response = response["answer"]
        except Exception as e:
            bot_response = f"Sorry, I couldn't process your question. ({e})"

        chat_history.append({"sender": "bot", "text": bot_response})

    return render_template("index.html", messages=chat_history)

if __name__=='__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
