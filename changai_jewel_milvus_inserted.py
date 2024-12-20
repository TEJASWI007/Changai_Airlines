import json
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
import uuid

client = MilvusClient(
    uri="https://in03-b9d74f3f1b5db46.serverless.gcp-us-west1.cloud.zilliz.com",
    token="50ddb73f2b67bbf0eacc1097934a14ac83f247c025112cae9352511c93afa70c3515148cd61976d4bec0d177704ea9bcb0573019",
)

collection_name = "CHANGAIJEWELchatbot"
client.load_collection(collection_name=collection_name)
print(f"Collection '{collection_name}' loaded successfully.")

with open("changai_jewel_chunking.json", "r") as file:
    data = json.load(file)

bgem3_embedding_function = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
sentence_transformer = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

start_index = 0
if start_index >= len(data):
    print(f"Starting index {start_index} exceeds data length {len(data)}.")
    exit()

for i, entry in enumerate(data[start_index:], start=start_index):
    full_content = entry.get("full_content", "")
    url = entry.get("URL", "")

    if full_content:
        bgem3_embedding = bgem3_embedding_function.encode_documents([full_content])
        bgem3_dense_embedding = (
            bgem3_embedding["dense"][0]
            if isinstance(bgem3_embedding["dense"], list)
            else bgem3_embedding["dense"]["array"][0]
        )
        bgem3_sparse_embedding = bgem3_embedding["sparse"]
        sentence_transformer_embedding = sentence_transformer.encode(full_content)

        print(f'Processed Entry {i + 1}/{len(data)}:')
        print(f'  - Length of BGEM3 Dense Embedding: {len(bgem3_dense_embedding)}')
        print(f'  - Length of HuggingFace Embedding: {len(sentence_transformer_embedding)}')

        unique_id = i + 1

        doc = {
            "primary_key": unique_id,
            "Sentence_Tranformer_Dense": sentence_transformer_embedding,
            "BGEM3_Dense": bgem3_dense_embedding,
            "BGEM3_Sparse": bgem3_sparse_embedding,
            "full_content": full_content,
            "URL": url,
        }

        try:
            response = client.upsert(collection_name=collection_name, data=[doc])
            print(f"Data inserted successfully. Primary Key: {unique_id}, Response: {response}")
        except Exception as e:
            print(f"Error inserting data for Primary Key: {unique_id}. Error: {e}")
    else:
        print(f"Entry {i + 1} skipped due to empty content.")

    with open("last_processed_index.txt", "w") as f:
        f.write(str(i + 1))
