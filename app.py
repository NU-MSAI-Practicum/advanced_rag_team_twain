import time
import random
import streamlit as st
from chatbot import get_response


st.title("Advanced RAG")
message = st.chat_message("assistant")
message.write("Hello there! How can I assist you today?")

# Sidebar
retrieval_selectbox = st.sidebar.selectbox(
    'Retrieval method',
    ('Sparse', 'Dense')
)

if retrieval_selectbox == 'Dense':
    text_splitter_chunk_size_selectbox = st.sidebar.selectbox(
        'Text Splitter Chunk Size',
        (1000, )
    )
    text_splitter_overlap_size_selectbox = st.sidebar.selectbox(
        'Text Splitter Overlap Size',
        (200, )
    )
    embedding_model_selectbox = st.sidebar.selectbox(
        'Embedding Model',
        ('Sentence Transformer', )
    )
    generator_selectbox = st.sidebar.selectbox(
        'Generator Model',
        ('Llama 3', )
    )

st.sidebar.write("Example Questions")
st.sidebar.write("When was the house at 3524 Redwing Ct, Naperville, IL 60564 last sold and for what price?")


# Streamed response emulator
def default_response_generator(prompt, retrieval_selectbox):
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


def response_generator(prompt, retrieval_selectbox):
    response = get_response(prompt, retrieval_selectbox)
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt, retrieval_selectbox))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
