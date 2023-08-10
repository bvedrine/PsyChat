
import os
import sys
import openai

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.document_loaders import GoogleDriveLoader


# https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety
openai.api_key = os.getenv("OPENAI_API_KEY")


# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# https://www.geeksforgeeks.org/how-to-use-sys-argv-in-python/
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader_txt = DirectoryLoader("extra_data/", glob="**/*.txt")
    loader_pdf = PyPDFDirectoryLoader("extra_data/")
    #loader_gdrive = GoogleDriveLoader(folder_id="16NBtR3fRFXtlHjuw_OQkon22PlIYwzhO",
    #                                  file_types=["document", "sheet"],
    #                                  recursive=False)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader_txt,
                                                                                                           loader_pdf])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader_txt, loader_pdf])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
