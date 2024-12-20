from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType,Collection,connections


connections.connect(uri="https://in03-b9d74f3f1b5db46.serverless.gcp-us-west1.cloud.zilliz.com",
                    token="50ddb73f2b67bbf0eacc1097934a14ac83f247c025112cae9352511c93afa70c3515148cd61976d4bec0d177704ea9bcb0573019")


fields = [
    FieldSchema(name="primary_key", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="Sentence_Tranformer_Dense", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="BGEM3_Dense", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="BGEM3_Sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name="full_content", dtype=DataType.JSON),
    FieldSchema(name="URL", dtype=DataType.JSON)
]

schema = CollectionSchema(fields=fields, description="changai jewel chatbot",enable_dynamic_field=True)

collection_name = "CHANGAIJEWELchatbot"
collection = Collection(name=collection_name, schema=schema, using="default")
print(f"Collection '{collection_name}' created.")

Sentence_Tranformer_Dense_index_params_vector = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}

BGEM3_dense_index_params_vector = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128},
}
BGEM3_sparse_index_params_vector = {
    "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "IP",
        "params": {"nlist": 128}
    }

collection.create_index("Sentence_Tranformer_Dense", Sentence_Tranformer_Dense_index_params_vector)
collection.create_index("BGEM3_Dense", BGEM3_dense_index_params_vector)
collection.create_index("BGEM3_Sparse", BGEM3_sparse_index_params_vector)

print(f"Index created on 'CohereDense' field in collection '{collection_name}'.")

collection.load()
print(f"Collection '{collection_name}' loaded into memory.")
print(collection.schema)
