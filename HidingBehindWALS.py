from google.colab import userdata, drive
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
from langchain_pinecone import PineconeVectorStore as LangChainPinecone

from vectorstore import setup_vectorstore

def setup_agent():
  index, embeddings = setup_vectorstore()
  vectorstore = LangChainPinecone(index, embeddings, text_key='text')

  llm = LangChainChatOpenAI(
      openai_api_key=userdata.get('OPENAI_KEY'),
      model_name='gpt-3.5-turbo',
      temperature=0.9
  )

  agent = RetrievalQA.from_chain_type(
      llm=llm,
      retriever=vectorstore.as_retriever()
  )
  return agent

class WALS():

  def __init__(self):
    self.agent = setup_agent()

  def TalkToWALS(self, query):
    preamble = """You are an expert on the World Atlas of Language Structures, also called WALS.
    Assume that any questions you are asked are referring to WALS, so if someone says Chatper 12
    they mean Chapter 12 in WALS, if they ask about verbal morphology they really mean what does
    WALS say about verbal morphology. Act like a helpful librarian who knows about WALS."""
    response = self.agent.invoke('\n'.join([preamble, query]))
    return response
