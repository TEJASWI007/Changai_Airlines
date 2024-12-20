from pymilvus import MilvusClient, Collection, connections
from pymilvus import AnnSearchRequest, RRFRanker
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from sentence_transformers import SentenceTransformer
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st

ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

def BGEM3_Embedding_dense(text):
    embedding = ef.encode_queries([text])
    embedding_of_dense = embedding["dense"]
    if isinstance(embedding_of_dense, list) and len(embedding_of_dense) > 0:
        return list(map(float, embedding_of_dense[0]))
    elif isinstance(embedding_of_dense, dict) and "array" in embedding_of_dense:
        return list(map(float, embedding_of_dense["array"][0]))
    return embedding_of_dense

def BGEM3_Embedding_sparse(text):
    embedding = ef.encode_queries([text])
    embedding_of_sparse = embedding["sparse"]
    return embedding_of_sparse

def HUGGING_FACE(text):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    huggingface_embedding = model.encode([text])
    return huggingface_embedding

def search(collection, input_vector1, input_vector2):
    search_param_1 = {
        "data": input_vector1,
        "anns_field": "Sentence_Tranformer_Dense",
        "param": {"metric_type": "COSINE", "params": {"nprobe": 10}},
        "limit": 5
    }
    request_1 = AnnSearchRequest(**search_param_1)

    search_param_2 = {
        "data": input_vector2,
        "anns_field": "BGEM3_Sparse",
        "param": {"metric_type": "IP", "params": {"nprobe": 10}},
        "limit": 5
    }
    request_2 = AnnSearchRequest(**search_param_2)

    reqs = [request_1, request_2]
    rerank = RRFRanker(k=60)
    res = collection.hybrid_search(reqs, rerank, limit=5, output_fields=['URL', 'full_content'])

    results = []
    for idx, hit in enumerate(res[0]):
        score = hit.distance
        entity = hit.entity
        url = getattr(entity, "URL", "No URL available")
        full_content = getattr(entity, "full_content", "No content available")
        results.append({"score": score, "url": url, "transformed_text": full_content})
    return results

def main():
    st.title("Law Bot")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(k=3)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask a question:"):
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        connections.connect(
            uri="https://in03-b9d74f3f1b5db46.serverless.gcp-us-west1.cloud.zilliz.com",
            token="50ddb73f2b67bbf0eacc1097934a14ac83f247c025112cae9352511c93afa70c3515148cd61976d4bec0d177704ea9bcb0573019",
        )
        collection = Collection("CHANGAIJEWELchatbot")

        input_vector1 = HUGGING_FACE(user_input)
        input_vector2 = BGEM3_Embedding_sparse(user_input)

        searching = search(collection, input_vector1, input_vector2)

        conversation_history = st.session_state.memory.load_memory_variables({})

        chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key="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",
            max_tokens=7500,
        )
        chain = load_qa_chain(llm=chat, verbose=True)
        input_documents = [
            Document(page_content=result['transformed_text'], metadata={"url": result['url']})
            for result in searching
        ]
        prompt = (
            """You have excellent knowledge about Changi Airport Group and Jewel Changi Airport.
            Please answer comprehensively based on these guidelines.
            """
        )
        response = chain.run(
            {
                'input_documents': input_documents,
                'question': f"{prompt}\n\n{conversation_history}\n\nQuestion: {user_input}"
            }
        )

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        st.session_state.memory.save_context({"input": user_input}, {"output": response})

if __name__ == "__main__":
    main()


