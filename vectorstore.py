from uuid import uuid4
from google.colab import userdata, drive
from pinecone import Pinecone, PodSpec, PineconeApiException
from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings


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
      "metadata": {"topic": f"Data about the {family} family"},
      "text": full_text,
    }]

    index.upsert(vector)