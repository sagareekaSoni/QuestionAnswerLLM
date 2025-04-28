## install
#  
#pip install llama-index-embeddings-huggingface

## import library

from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader,StorageContext,Settings
from llama_index.llms.ollama import Ollama
from elasticsearch import Elasticsearch
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from elasticsearch import AsyncElasticsearch

## Load data
document = SimpleDirectoryReader('./Data/Troubleshooting Guides').load_data()


## Set up Elasticsearch vector store
es= AsyncElasticsearch("http://localhost:9200")

Settings.llm = Ollama(model="llama2", request_timeout=120.0)
# load embed model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

vc = ElasticsearchStore(index_name='qallm',es_client=es,embedding_dim=384,vector_field="embedding",distance_strategy="COSINE")
sc=StorageContext.from_defaults(vector_store=vc)

index = VectorStoreIndex.from_documents(document,embed_model=Settings.embed_model,storage_context=sc)

# Define a question
question = "what is the solution to Applications that use Novell NetBIOS (such as Windows 95) cannot connect to servers over a router. Clients cannot connect to servers on the same LAN."

# Retrieve relevant context from the index
query_engine = index.as_query_engine(llm=Settings.llm)

# Get answer from LLaMA
context = query_engine.query("what is token ring network ?").response
print(context)


