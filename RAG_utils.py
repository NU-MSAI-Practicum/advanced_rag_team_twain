
from CreateDocuments import load_documents

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




def create_chroma_db(doc_path='documents/train_size_1000_overlap_200_documents.pkl', collection_name="rag_demo_collection"):
    """
    Create a Chroma database from a list of documents.
    Initializes the databse with chromadb (instead of langchain_community.vectorstores.Chroma) to enable more functionalities (e.g., peek()).

    Args:
        doc_path (string): The path to the list of documents.
        collection_name (string): The name of the collection in the database.

    Returns:
        The langchain Chroma object.
    """

    docs = load_documents(doc_path)
    print("Number of documents:", len(docs))
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    persistent_client = chromadb.PersistentClient()
    persistent_client.delete_collection(collection_name)
    collection = persistent_client.get_or_create_collection(collection_name)

    # Make langchain Chroma object from the collection
    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    langchain_chroma.add_documents(docs)
    print("There are", langchain_chroma._collection.count(), "in the collection")

    return langchain_chroma


def format_docs(docs):
    """
    Format a list of documents into a single string.

    Args:
        docs (list): A list of documents.

    Returns:
        A single string containing the content of all the documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


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

def gen_text_hf_local(lm, tokenizer, prompt_template, context, question):
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

    prompt_text = prompt_template.format(context=context, question=question)

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

def gen_text_hf_api(lm_name, prompt_text):
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
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03,
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

# def full_RAG(db, lm_name, prompt_template, question):
#     lm, tokenizer = load_lm_and_tokenizer(lm_name)
#     docs = db.query(question)
#     context = format_docs(docs)
#     return rag_chain(lm, tokenizer, prompt_template, context, question)

# def RAG_with_context(lm_name, prompt_template, question, context):
#     lm, tokenizer = load_lm_and_tokenizer(lm_name)
#     return rag_chain(lm, tokenizer, prompt_template, context, question)

if __name__ == '__main__':
    # create_chroma_db()
    pass