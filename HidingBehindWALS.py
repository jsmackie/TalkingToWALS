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
from langchain_pinecone import Pinecone as LangChainPinecone

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
    #Note: this might be too large, and maybe should be split again with RecursiveTextSplitter
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
      print('Pinecone exception occured. File is bigger than 2MB! Igoring.')

def setup_vectorstore(split_type='json', use_existing_index=True):
    #WALS documents are available in text, html, and json formats
    #Chunking plain text seems to give the worst results, likely becaues it's blind to context right now
    #html gives better results, because WALS is already structured by headings
    #json gives best results, because each text blob can be augemented with some metadata
    #this allows every paragraph in a section to carry information about the section name and topic

    split_type = split_type

    text_embedding_model = 'text-embedding-ada-002'
    #TODD: experiment with other embedding models
    embeddings = LangChainOpenAIEmbeddings(
        model = text_embedding_model,
        openai_api_key=userdata.get('OPENAI_KEY'))

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
      pass #no extra formatting required

    pc = Pinecone(api_key=userdata.get('PINECONE_TOKEN'))
    index_name = 'starter-index'

    if use_existing_index:
      index = pc.Index(index_name)
    else:
      #See https://docs.pinecone.io/docs/manage-indexes#create-a-pod-based-index
      pc.delete_index(index_name)
      pc.create_index(name=index_name, dimension=1536, metric="cosine", spec=PodSpec(environment="gcp-starter"))
      index = pc.Index(index_name)
      #Get table of contents and other general metadata about WALS
      with open('/content/drive/MyDrive/WALS/chapter_metadata.csv', encoding='utf-8') as file:
        file_content = file.read()
      #The metata below is for human readability in the Pincone UI, it is not provided to the model
      chapter_metadata = load_chapter_metadata('/content/drive/MyDrive/WALS/chapter_metadata.csv')

      if split_type == 'text':
        text_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/', chapter_metadata, splitter, embeddings, subdirectory='chapters_html')
      elif split_type == 'html':
        html_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/chapters_html', chapter_metadata, splitter, embeddings)
      elif split_type == 'json':
        json_splitter_create_and_upsert_vectors(index, '/content/drive/MyDrive/WALS/chapters_json', chapter_metadata, embeddings)

    vectorstore = LangChainPinecone(index, embeddings, text_key='text')

    return vectorstore

def TalkToWALS():

    vectorstore = setup_vectorstore()

    llm = LangChainChatOpenAI(
        openai_api_key=userdata.get('OPENAI_KEY'),
        model_name='gpt-3.5-turbo',
        temperature=0.8
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa
