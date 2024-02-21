from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
import sys

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#**Step 1: Load the PDF File from Data Path****
loader=DirectoryLoader('/workspace/assets/doc/',
                       glob="0072.pdf",
                       loader_cls=PyPDFLoader)

documents=loader.load()

# print(documents)

#***Step 2: Split Text into Chunks***
text_splitter=RecursiveCharacterTextSplitter(
                                             chunk_size=500,
                                             chunk_overlap=50)

text_chunks=text_splitter.split_documents(documents)

print(len(text_chunks))

#**Step 3: Load the Embedding Model***
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cuda'})

#**Step 4: Convert the Text Chunks into Embeddings and Create a FAISS Vector Store***
vector_store=FAISS.from_documents(text_chunks, embeddings, use_gpu=True)

##**Step 5: Find the Top 3 Answers for the Query***
query="Performance Bond"
docs = vector_store.similarity_search(query)

print(docs)

llm=CTransformers(model="/workspace/models/llama-2-7b-chat.Q4_K_M.gguf",
                  model_type="llama",
                  config={'max_new_tokens':128,
                          'temperature':0.01},
                  callbacks=[StreamingStdOutCallbackHandler()],
                  )

template="""Use the following pieces of information to answer the user's question.
If you dont know the answer just say you know, don't try to make up an answer.

Context:{context}
Question:{question}

Only return the helpful answer below and nothing else
Helpful answer
"""

qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

chain = RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=vector_store.as_retriever(search_kwargs={'k': 2}),
                                   return_source_documents=True,
                                   chain_type_kwargs={'prompt': qa_prompt})

result = chain({'query':"please tell me about performance bond"})


print(result)