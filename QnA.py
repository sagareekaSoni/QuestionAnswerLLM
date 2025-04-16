# install
# 

#pip install llama-index-embeddings-huggingface
# import library

from llama_index.core import Document, VectorStoreIndex, ServiceContext, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

document = SimpleDirectoryReader('./data').load_data()
# Set up Elasticsearch vector store
es_client = ElasticsearchStore(index_name='Q_A_llm',
                               es_url="http://localhost:9200",
                               embedding_dim=384,
                               vector_field="embedding",
                               distance_strategy="COSINE")

#em_model = 
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

index = VectorStoreIndex.from_documents(document, vector_store=es_client)




import ollama

query_engine = index.as_query_engine()
query_engine.query("Cisco Catalyst 9600 Series Switches offer")







