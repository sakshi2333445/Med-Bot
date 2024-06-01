from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS  
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA 
import chainlit as cl
import re
import nltk
from nltk.corpus import words  # Import a list of English words

nltk.download('words')

DB_FAISS_PATH = "vectorstores/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question. 
If you don't know the answer , please just say that you don't know the answer , don't try to make up an answer.

Context:{context}
Question :{question}
Only returns the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """Prompt template for QA retrieval for each vector stores
    """
    
    prompt = PromptTemplate(template=custom_prompt_template,input_variables=["context","question"]) 
    return prompt

def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_token = 512, #max no of new token (output length)
        temperature = 0.5 #randomness of model
    )
    
    return llm

def retrieval_qa_chain(llm,prompt,db):
    qa_chain = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type = "stuff",
        retriever = db.as_retriever(search_kwargs={"k": 2}),    # k is no of searches 
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents = True,                         # use only custom knowledge provided

    )
    
    return qa_chain


def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs = {'device':'cuda'})
    
    db = FAISS.load_local(DB_FAISS_PATH,embeddings,allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm,qa_prompt,db)
    
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query':query})
    return response


### Chainlit ###

#to start the bot
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content  = 'Starting the bot...')
    await msg.send()
    #update the screen 
    msg.content = "Hi, Welcome to the Medical Bot! What is your query??"
    await msg.update()
    cl.user_session.set("chain",chain)

#when user asks the question
@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    user_input = message.content.strip()

    if not is_valid_query(user_input):  # Check if input is valid
        await cl.Message(content="Nothing matched. Please enter a valid query.").send()
        return

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer = True,answer_prefix_tokens = ["FINAL ANSWER"]
    )
    
    cb.answer_reached = True
    res = await chain.acall(message.content,callbacks = [cb])
    answer = res["result"]
    sources = res["source_documents"]
    

    if not answer:
        await cl.Message(content="No information found for your query.").send()
        return
    
    '''To add source information in chatbot's output'''
    # if sources :
    #     answer += f"\n Sources:" + str(sources)
    # else :
    #     answer += f"\n No sources found "
        
    await cl.Message(content=answer).send()



def is_valid_query(query):

    # Check if the query is empty or contains only whitespace
    if not query or query.isspace():
        return False
    
    # Check if the query contains only special characters
    if not re.search(r'[a-zA-Z0-9]', query):
        return False
    
    
    return True    
