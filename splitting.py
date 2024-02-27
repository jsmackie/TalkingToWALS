import os
import json
from uuid import uuid4
import tiktoken
from pinecone import Pinecone, PodSpec, PineconeApiException
from langchain.text_splitter import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter

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