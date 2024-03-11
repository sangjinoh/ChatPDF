import time
import pprint

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

from langchain.embeddings import HuggingFaceEmbeddings

model_name = 'jhgan/ko-sbert-nli'
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader('assets/doc/운전자보험상품약관.pdf')
pages = loader.load_and_split()
print('[+] doc load done!!')

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5, length_function=tiktoken_len)
docs = text_splitter.split_documents(pages)

print('[+] Split Done!!')

from langchain.vectorstores import Chroma

# save to disk
saved_db = Chroma.from_documents(docs, ko, persist_directory='chroma_db')
loaded_db = Chroma(persist_directory="chroma_db", embedding_function=ko)

print('[+] Embedding done!!')

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp

model_path = "assets/models/llama-2-7b-chat.Q4_K_M.gguf"

llama = LlamaCpp(
    model_path=model_path,
    streaming=True,
)

print('[+] Ready to LLM!!')

from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llama,
                                 chain_type="stuff",
                                 retriever=loaded_db.as_retriever(
                                     search_type="mmr",
                                     search_kwargs={'k':3, 'fetch_k': 10}
                                 ),
                                 return_source_documents=True
                                 )

print('[+] Ready to chain\n')

query = "소비자가  반드시  알아두어야  할 상품의  주요 특성은 뭐야?"

start_time = time.time()
print(time.ctime(start_time))
result = qa(query)
print("\nElapsed Time:", time.ctime(time.time()-start_time))

pprint.pprint(result)


