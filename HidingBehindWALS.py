import os
import json
from uuid import uuid4

import tiktoken

from google.colab import userdata, drive

from pinecone import Pinecone, PodSpec, PineconeApiException

from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as LangChainPinecone

def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def load_chapter_metadata(path):
  with open(os.path.join(path)) as f:
    headers = f.readline().strip().split(',')
    lines = [line.strip().split(',') for line in f]
  chapter_metadata = dict()
  for i,line in enumerate(lines):
    chapter_metadata[i] = {'chapter number': i+1,
                          'title': line[headers.index('Title')],
                          'authors': line[headers.index('Contributors')],
                          'linguistic subfield': line[headers.index('Area')]}
  return chapter_metadata

def text_splitter_create_and_upsert_vectors(index,
                                            path,
                                            chapter_metadata,
                                            text_splitter,
                                            embeddings,
                                            batch_limit=100,
                                            subdirectory='chapters'):
  batch_limit = batch_limit
  texts = []
  metadatas = []
  for i,file in enumerate(os.listdir(os.path.join(path, subdirectory))):
    metadata = chapter_metadata[i]
    with open(os.path.join(path, subdirectory, file), mode='r') as f:
      text = ''.join([line for line in f])
    record_texts = text_splitter.split_text(text)
    record_metadatas = [{"chunk": j, "text": text, **metadata}
                          for j, text in enumerate(record_texts)]
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    if len(texts) >= batch_limit:
      ids = [str(uuid4()) for _ in range(len(texts))]
      embeds = embeddings.embed_documents(texts)
      index.upsert(vectors=zip(ids, embeds, metadatas))
      texts = []
      metadatas = []

def html_splitter_create_and_upsert_vectors(index,
                                            path,
                                            chapter_metadata,
                                            html_splitter,
                                            embeddings):
  batch_limit=100
  texts = list()
  metadatas = list()
  for i,file in enumerate(os.listdir(path)):
    #print(file)
    if not file.endswith('html'):
      continue
    html_splits = html_splitter.split_text_from_file(os.path.join(path, file))
    #Note: this might be too large, and maybe should be split again with RecursiveTextSplitter?
    ids = [str(uuid4()) for _ in range(len(html_splits))]
    embeds = embeddings.embed_documents([split.page_content for split in html_splits])
    #metadatas = [split.metadata for split in html_splits]
    metadatas = {key:value for key,value in chapter_metadata[i].items()}
    for split in html_splits:
      metadatas.update(split.metadata)
    record_metadatas = [{"chunk": j, "text": split.page_content, **metadatas}
                          for j, split in enumerate(html_splits)]
    index.upsert(vectors=zip(ids, embeds, record_metadatas))

def json_splitter_create_and_upsert_vectors(index, path, chapter_metadata, embeddings):
  texts = list()
  for i,file in enumerate(sorted(os.listdir(path))):
    if not file.endswith('.json'):
      continue
    print(i, file)
    metadatas = {key:value for key,value in chapter_metadata[i].items()}
    text_preamble = ''.join(['Hints: ',
                            ','.join([f'{k}={v}' for (k,v) in metadatas.items()]),
                            '\n'])
    with open(os.path.join(path, file), encoding='utf-8', mode='r') as f:
      json_file = json.load(f)
    for blob in json_file:
      chapter_metadata[i]['section'] = list()
      try:
        chapter_metadata[i]['section'].append(blob['section'].strip())
      except KeyError:
        chapter_metadata[i]['section'].append('1. Introduction')
      for p in blob['paragraphs']:
        texts.append('\n'.join([text_preamble, p['text']]))
    embeds = embeddings.embed_documents(texts)

    ids = [str(uuid4()) for _ in range(len(texts))]
    records = [{"chunk": j, "text": text, **metadatas}
                          for j, text in enumerate(texts)]
    try:
      index.upsert(vectors=zip(ids, embeds, records))
    except PineconeApiException:
      print('Pinecone exception occured. File is bigger than 2MB! Ignoring.')

def upsert_wals_data(index, embeddings):
  #Get table of contents and other general metadata about WALS
  with open('/content/drive/MyDrive/WALS/chapter_metadata.csv', encoding='utf-8') as file:
    file_content = file.read()
  embeds = embeddings.embed_documents([file_content])
  vector = [{
      "id":str(uuid4()),
      "values": embeds[0],
      "metadata": {"topic": "table of contents"}
    }]
  index.upsert(vector)

def upsert_language_data(index, embeddings):
  with open('/content/drive/MyDrive/WALS/languages.csv') as f:
    headers = f.readline().strip().split(',')
    rows = [row.strip().split(',') for row in f]
  blobs = list()
  for row in rows:
    blobs.append({headers[j]:value for (j,value) in enumerate(row)})
  #There are ~3000 languages, which makes too many individual records
  #try grouping them by family

  families = list(set([language['Family'] for language in blobs]))
  for family in families:
    full_text = list()

    family_matches = [language for language in blobs if language['Family'] == family]
    genera = list(set([language['Genus'] for language in family_matches]))
    full_text.append(f'Language Family: {family} has {len(genera)} subdivisions/groups/genera/etc.\n')

    for genus in genera:
      full_text.append(f'- Genus: {genus} includes these languages: \n-- ')
      genus_matches = [language for language in family_matches if language['Genus'] == genus]

      language_text = list()
      for language in genus_matches:
        language_text.append('{} ISO code {} spoken in {}'.format(language['Name'], language['ISO639'], language['Macroarea'] ))
      language_text = '\n-- '.join(language_text)
      full_text.append(language_text)

    full_text = ''.join(full_text)
    #print(full_text)
    embeds = embeddings.embed_documents([full_text])
    vector = [{
      "id":str(uuid4()),
      "values": embeds[0],
      "metadata": {"topic": f"Data about the {family} family"}
      "text": full_text,
    }]

    index.upsert(vector)

def get_splitter(split_type='html'):
  #WALS documents are available in text, html, and json formats
  #Chunking plain text seems to give the worst results, likely becaues it's blind to context right now
  #html gives better results, because WALS is already structured by headings
  #json gives best results, because each text blob can be augemented with some metadata
  #this allows every paragraph in a section to carry information about the section name and topic

  if split_type == 'text':
    tokenizer = tiktoken.get_encoding('p50k_base')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
        )
  elif split_type == 'html':
    headers = [("h2", "topic"), ("h5", "introduction")]
    splitter = HTMLHeaderTextSplitter(headers)
  elif split_type == 'json':
    return None

  return splitter

def setup_vectorstore(make_new_index = False,
                      upsert_metadata = False,
                      upsert_chapter_data = False,
                      index_name='starter-index',
                      split_type = 'json',
                      text_embedding = 'text-embedding-ada-002'):

  pc = Pinecone(api_key=userdata.get('PINECONE_TOKEN'))
  embeddings = LangChainOpenAIEmbeddings(
      model =  text_embedding,
      openai_api_key=userdata.get('OPENAI_KEY'))

  if make_new_index:
    #https://docs.pinecone.io/docs/manage-indexes#create-a-pod-based-index
    pc.delete_index(index_name)
    pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=PodSpec(environment="gcp-starter"))
  index = pc.Index(index_name)

  if upsert_metadata:
    upsert_wals_data(index, embeddings)
    upsert_language_data(index, embeddings)


  if upsert_chapter_data:
    chapter_metadata = load_chapter_metadata('/content/drive/MyDrive/WALS/chapter_metadata.csv')
    splitter = get_splitter(split_type)
    if split_type == 'text':
      text_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/', chapter_metadata, splitter, embeddings, subdirectory='chapters_html')
    elif split_type == 'html':
      html_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/chapters_html', chapter_metadata, splitter, embeddings)
    elif split_type == 'json':
      json_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/chapters_json', chapter_metadata, embeddings)

  return index, embeddings

def setup_agent():
  index, embedding = setup_vectorstore()
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
    print(response['result'])
