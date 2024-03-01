
import chainlit as cl
import os
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv, dotenv_values
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import copy
from typing import Optional

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict



id = "1st convo"
pc = Pinecone(os.getenv("PINECONE_API_KEY"))
index = pc.Index("chatbotgpt")

# prompt template for the chatbot
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

PROMPT = """ 

Rôle et Objectif : EDN GPT est un assistant dédié aux étudiants en médecine préparant l'examen de l'EDN. Il se concentre uniquement sur les informations contenues dans les documents spécifiques fournis sous forme de PDF, utilisés comme référence pour l'examen de l'EDN.
Il répond de manière structurée avec une introduction, plusieurs points points avec des titres puis une conclusion.

Contraintes : EDN GPT doit utiliser uniquement les informations pertinentes pour l'examen de l'EDN, qui est la référence pour les étudiants. Il ne doit ABSOLUMENT jamais proposer de se référer à un médecin ni à d'autres directives car les questions posées par les étudiants sont uniquement dans le but de trouver des informations sur des cours. Il ne s'agit pas de questions pour obtenir des conseils médicaux.
Tu dois te rappeler de la question d'avant pour fournir des précisions supplémentaires si besoin.
Il ne doit JAMAIS rappeler qu'il s'agit de données issues des documents fournis ni de se référer à un médecin.

Directives : Il doit être précis et donner une réponse complète et détaillé. Il doit demander des clarifications si la requête de l'utilisateur est vague ou si le matériel de référence n'est pas assez spécifique, et viser à fournir des informations précises et pertinentes.

Personnalisation : EDN GPT offre des réponses détaillées et adaptées aux besoins des étudiants en médecine se préparant pour l'EDN, avec patience et profondeur dans ses explications. Il répond exclusivement en français.

"""
client = OpenAI(
    # This is the default and can be omitted
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key = os.getenv("OPENAI_API_KEY"),
)

# thread_id = thread_LKjnEX1rKhYmYDSI9KMJ88Fw


map = {}

def embed(input):
    return client.embeddings.create(
        input=input,
        model="text-embedding-3-small"
    ).data[0].embedding
#.data[0].embedding


def split_text(text):
    # Split the text into chunks using Langchain's CharacterTextSplitter
    text_splitter = CharacterTextSplitter( 
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # Convert the chunks of text into embeddings to form a knowledge base
    #embeddings = OpenAIEmbeddings()
    #knowledgeBase = FAISS.from_texts(chunks, embeddings)
    
    return chunks

def chunk_pdf(file):
    # file is the path to the PDF file
    loader = PyPDFLoader(file)
    docs = loader.load()
    all_pages_chunks = []
    pg=0
    for page in docs:
        splited_page = split_text(page.page_content)
        cur=0
        for chunk in splited_page:
            temp = copy.deepcopy(page)
            temp.page_content = chunk
            temp.metadata['id'] = "page_"+str(pg)+"_chunk_"+str(cur)
            temp.metadata['text'] = chunk
            #print(temp.metadata['id'])
            cur+=1
            all_pages_chunks.append(temp)
            #print(all_pages_chunks[-1].metadata['id'])
        pg +=1
    return all_pages_chunks

def embed_pdf(file):
    # file is the path to the PDF file
    docs = chunk_pdf(file)
    for i in range(len(docs)):
        docs[i].metadata['embedding'] = embed(docs[i].page_content)
    #docs[10].metadata['embedding'] = embed(docs[10].page_content)
    return docs

def push_db(file):
    # Push the documents to the database
    chunks = chunk_pdf(file)
    vectors = []
    for chunk in chunks:
        vectors.append({"id": chunk.metadata['id'], "values": embed(chunk.page_content),"metadata": chunk.metadata})
    index.upsert(vectors=vectors)

def topk(question, k=10):
    # Get the top k documents that are most relevant to the question
    question_embedding = embed(question)
    results = index.query(vector=question_embedding, include_metadata=True, top_k=k)
    #getting the texts of the top k documents
    texts = []
    for result in results.matches:
        texts.append(result.metadata['text'])
    context_text = "\n\n---\n\n".join([text for text in texts])
    return context_text

def answer(question):
    context = topk(question)
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "system", "content": "You are a helpful research assistant."}, {"role": "user", "content": prompt}]
    )
    return (prompt,response.choices[0].message.content)


def main(question):
    all_pages_chunks = chunk_pdf("options.pdf")
    
    for page in all_pages_chunks:
        map[page.metadata['id']] = page.page_content

    #question = "what will happen to an airline who bought millions of gallons of jet fuel and decided to hedge"

    ans = answer(question)
    return ans[1]


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)

    setup_runnable()

@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    # do some work
    await cl.sleep(2)

    answer = main(message.content)
    msg.content = answer

    await msg.update()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None