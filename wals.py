from langchain_pinecone import PineconeVectorStore as LangChainPinecone
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
from langchain.chains import RetrievalQA
from HidingBehindWALS import setup_vectorstore


class WALSBuilder:

    def __init__(self, open_ai_key, personality=None):
        self.agent = self.setup_agent(open_ai_key)
        self.set_agent_personality(personality)

    def set_agent_personality(self, personality=None):
        self.preamble = 'You are an expert on the World Atlas of Language Structures, also called WALS.'
        if personality == 'sagan':
            self.preamble += """
            Your sole goal is to help the user understand more about what's in WALS, and to inspire them to get excited
            about linguistics and language typology. You can act like a combination of David Attenborough and Carl Sagan, but
            with a passion for language instead.
            """
        elif personality == 'librarian':
            self.preamble += """
            Assume that any questions you are asked are referring to WALS, so if someone says Chapter 12
            they mean Chapter 12 in WALS, if they ask about verbal morphology they really mean what does
            WALS say about verbal morphology. Act like a helpful librarian who knows about WALS.
            """

    def TalkToWALS(self, query):
        response = self.agent.invoke('\n'.join([self.preamble, query]))
        print(response['result'])

    def setup_agent(self, api_key):
        index, embeddings = setup_vectorstore(make_new_index=False,
                                              upsert_metadata=False,
                                              upsert_chapter_data=False)
        vectorstore = LangChainPinecone(index, embeddings, text_key='text')

        llm = LangChainChatOpenAI(
            openai_api_key=api_key,
            model_name='gpt-3.5-turbo',
            temperature=0.9
        )

        agent = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )
        return agent

    def interact(self, message, history):

        response = self.agent.invoke(''.join([self.preamble, message]))
        text = response['result']
        return text
