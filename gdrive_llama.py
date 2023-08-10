

# https://betterprogramming.pub/how-to-build-your-own-custom-chatgpt-with-custom-knowledge-base-4e61ad82427e


import pickle
import os
import openai
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from llama_index import VectorStoreIndex, SimpleDirectoryReader, download_loader, \
    StorageContext, load_index_from_storage, ListIndex
from langchain.indexes import VectorstoreIndexCreator

from authorize_gdocs import authorize_gdocs

openai.api_key = os.getenv("OPENAI_API_KEY")

# https://medium.com/@vanillaxiangshuyang/tailor-chatgpt-to-your-needs-with-custom-knowledge-bases-e75e8306960
# authorize or download latest credentials
authorize_gdocs()

# https://gpt-index.readthedocs.io/en/latest/getting_started/customization.html
# https://llamahub.ai/l/google_docs
GoogleDocsReader = download_loader('GoogleDocsReader')
gdoc_ids = ['1filndDjFJGWjqQJzan0RMzjLNnPnv90BQCIK12LPb3M']
loader = GoogleDocsReader()
documents = loader.load_data(document_ids=gdoc_ids)








#index = VectorStoreIndex.from_documents(documents)
index = ListIndex.from_documents(documents)

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine()


prompt = input("Type prompt: ")
response = query_engine.query(prompt)
print(response)

index.storage_context.persist(persist_dir="./storage")




