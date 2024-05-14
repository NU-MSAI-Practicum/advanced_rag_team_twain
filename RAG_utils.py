from CreateDocuments import load_chunks

# embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma
from langchain_community.vectorstores import Chroma
import chromadb

# transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# HF API
from langchain_community.llms import HuggingFaceHub


def list_chroma_collections():
    """
    List all collections in the Chroma database.

    Returns:
    A list of collection names.
    """
    persistent_client = chromadb.PersistentClient()
    return persistent_client.list_collections()


def create_chroma_db(chunks_path='chunks/train_size_1000_overlap_200.pkl', collection_name="rag_demo_collection", embedding_model="all-MiniLM-L6-v2"):
    """
    Create a Chroma database from a list of chunks.
    Initializes the databse with chromadb (instead of langchain_community.vectorstores.Chroma) to enable more functionalities (e.g., peek()).

    Args:
        chunks_path (string): The path to the chunks.
        collection_name (string): The name of the collection in the database.

    Returns:
        The langchain Chroma object.
    """

    docs = load_chunks(chunks_path)
    #print("Number of documents:", len(docs))
    
    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

    persistent_client = chromadb.PersistentClient()
    
    # Clear the collection if it exists
    try:
        persistent_client.delete_collection(collection_name)
    except Exception as e:
        print(f"Error deleting collection: {e}")


    collection = persistent_client.get_or_create_collection(collection_name)

    # Make langchain Chroma object from the collection
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    # Add documents to the collection in batches
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        langchain_chroma.add_documents(docs[i:i + batch_size])

    #print("There are", langchain_chroma._collection.count(), "in the collection")

    return langchain_chroma

def access_chroma_db(collection_name, embedding_model="all-MiniLM-L6-v2"):
    """
    Accesses an existing Chroma collection from a database by collection name.
    
    Args:
    - collection_name (str): The name of the collection to access.

    Returns:
    - Chroma: A Chroma object linked to the specified collection.
    """

    persistent_client = chromadb.PersistentClient()
    try:
        # Try to get the existing collection
        collection = persistent_client.get_collection(collection_name)
    except Exception as e:
        print(f"Error accessing collection {collection_name}: {e}")
        return None


    embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)

    # Create a Chroma object using the existing collection
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    #print("Accessed collection:", collection_name)
    return langchain_chroma

def format_docs(docs):
    """
    Format a list of Chroma documents into a single string.

    Args:
        docs (list): A list of documents.

    Returns:
        A single string containing the content of all the documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)

def format_contexts(contexts):
    """
    Format a list of contexts into a single string.

    Args:
        contexts (list): A list of contexts.

    Returns:
        A single string containing all the contexts.
    """
    return "\n\n".join(contexts)


def configure_and_load_model(lm_name, config_updates=None, device_name="cpu"):
    """
    Configure and load the language model.

    Args:
        lm_name (string): The name of the language model.
        config (transformers.PretrainedConfig): The configuration.
        device_name (string): The device to use.

    Returns:
        The language model.
    """
   

    config = AutoConfig.from_pretrained(lm_name)
    # Update model configs
    for key, value in config_updates.items():
        setattr(config, key, value)

    device = torch.device(device_name)
    lm = AutoModelForCausalLM.from_pretrained(lm_name, config=config).to(device)
    return lm

def load_lm_and_tokenizer(lm_name, config_updates=None, device_name="cpu"):
    """
    Gets the language model and tokenizer from Hugging Face.

    Args:
        lm_name (string): The name of the language model.
        config_updates (dict): The configuration updates.
        device (string): The device to use.

    Returns:
        The language model and tokenizer.
    """
    lm = configure_and_load_model(lm_name, config_updates, device_name)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    # Set the pad token to the eos token if it is None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return lm, tokenizer

def format_prompt(prompt_template, context, question):
    """
    Format the prompt with the context and question.

    Args:
        prompt_template (string): The prompt template.
        context (string): The context.
        question (string): The question.

    Returns:
        The formatted prompt.
    """
    return prompt_template.format(context=context, question=question)
                                  

def gen_text_hf_local(lm, tokenizer, prompt_text):
    """
    Generate text using the RAG model.
    Manually chains the context-question->prompt, and model for generation.

    Args:
        lm (torch.nn.Module): The language model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        prompt_template (string): The prompt template.
        context (string): The context.
        question (string): The question.

    Returns:
        The generated text.
    """

    #prompt_text = format_prompt(prompt_template, context, question)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    # Pass to model and get output tensors
    outputs = lm(**inputs)
    #print(lm)
    #print(outputs)

    predicted_token_ids = outputs.logits.argmax(dim=-1).squeeze()
    # Check if the result needs to be converted from tensor to list
    if isinstance(predicted_token_ids, torch.Tensor):
        predicted_token_ids = predicted_token_ids.tolist()  # Convert tensor to list
    # Decode output tensor to string
    generated_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
    return generated_text

def gen_text_hf_api(lm_name, prompt_text, temp=0.1, top_k=30, rep_pen=1.03):
    """
    Generate text using the Hugging Face Hub.

    Args:
        lm_name (string): The name of the language model.
        prompt_text (string): The prompt text.

    Returns:
        The generated text.
    """
    lm = HuggingFaceHub(
        repo_id=lm_name,
        #repo_id="microsoft/DialoGPT-medium",
        task="text-generation",
        huggingfacehub_api_token = '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',
        model_kwargs={
            "max_new_tokens": 250,
            "top_k": top_k,
            "temperature": temp,
            "repetition_penalty": rep_pen,
        },
    )
    
    generated_text = lm.invoke(prompt_text)[len(prompt_text):]
    return generated_text


def rag_chainV2(lm, tokenizer, prompt_template, context, question):
    prompt_text = prompt_template.format(context=context, question=question)
    # Tokenize the prompt text to prepare for model input
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    #print(len(inputs['input_ids']))

    outputs = lm.generate(
        inputs["input_ids"], 
        max_new_tokens=250, 
        top_k=30,
        temperature=0.1,
        repetition_penalty=1.03,
    )

    # Decode outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

import ollama
def gen_text_ollama_user_only(prompt_text, options=None):
    """
    Generate text using Ollama.

    Args:
        prompt_text (string): The prompt text.
        options (dict): The options for the Ollama model.
    Returns:
        The generated text (string).
    """
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': prompt_text}], options=options)
    return response['message']['content']

def gen_text_ollama(sys_msg, user_msg, model='llama3', options=None):
    """
    Generate text using Ollama.

    Args:
        sys_msg (string): The system prompt.
        user_msg (string): The user prompt.
        options (dict): The options for the Ollama model.
    Returns:
        The generated text (string).
    """
    response = ollama.chat(model=model, messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}], options=options)
    return response['message']['content']

'''Note: options for Ollama:
"options": {
    "num_keep": 5,
    "seed": 42,
    "num_predict": 100,
    "top_k": 20,
    "top_p": 0.9,
    "tfs_z": 0.5,
    "typical_p": 0.7,
    "repeat_last_n": 33,
    "temperature": 0.8,
    "repeat_penalty": 1.2,
    "presence_penalty": 1.5,
    "frequency_penalty": 1.0,
    "mirostat": 1,
    "mirostat_tau": 0.8,
    "mirostat_eta": 0.6,
    "penalize_newline": true,
    "stop": ["\n", "user:"],
    "numa": false,
    "num_ctx": 1024,
    "num_batch": 2,
    "num_gqa": 1,
    "num_gpu": 1,
    "main_gpu": 0,
    "low_vram": false,
    "f16_kv": true,
    "vocab_only": false,
    "use_mmap": true,
    "use_mlock": false,
    "rope_frequency_base": 1.1,
    "rope_frequency_scale": 0.8,
    "num_thread": 8
  }
Source: https://github.com/ollama/ollama/blob/main/docs/api.md
'''


if __name__ == '__main__':
    # create_chroma_db()
    pass