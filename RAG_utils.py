
from CreateDocuments import load_documents

# embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Chroma
from langchain_community.vectorstores import Chroma
import chromadb

# transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # pip install bitsandbytes




def create_chroma_db(doc_path='documents/train_size_1000_overlap_200_documents.pkl', collection_name="rag_demo_collection"):

    docs = load_documents(doc_path)
    print("Number of documents:", len(docs))
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    persistent_client = chromadb.PersistentClient()
    persistent_client.delete_collection(collection_name)
    collection = persistent_client.get_or_create_collection(collection_name)

    langchain_chroma = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embedding_model,
    )

    langchain_chroma.add_documents(docs)
    print("There are", langchain_chroma._collection.count(), "in the collection")

    return langchain_chroma


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def load_lm_and_tokenizer(lm_name):

    device = torch.device("cpu")
    lm = AutoModelForCausalLM.from_pretrained(lm_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    return lm, tokenizer

def rag_chain(lm, tokenizer, prompt_template, context, question):

    prompt_text = prompt_template.format(context=context, question=question)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    # Pass to model and get output tensors
    outputs = lm(**inputs)
    #print(lm)
    #print(outputs)

    predicted_token_ids = outputs.logits.argmax(dim=-1).squeeze()  # Use squeeze() to reduce dimensions
    # Check if the result needs to be converted from tensor to list
    if isinstance(predicted_token_ids, torch.Tensor):
        predicted_token_ids = predicted_token_ids.tolist()  # Convert tensor to list
    # Decode the output tensors to human-readable text
    generated_text = tokenizer.decode(predicted_token_ids, skip_special_tokens=True)
    return generated_text

def rag_chainV2(lm, tokenizer, prompt_template, context, question):
    prompt_text = prompt_template.format(context=context, question=question)
    # Tokenize the prompt text to prepare for model input
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024)
    print(len(inputs['input_ids']))
    # Pass to model and get output tensors
    outputs = lm.generate(
        inputs["input_ids"], 
        max_length=50,  # Specify your max length for the generated text
        num_beams=5,  # Number of beams
        early_stopping=True  # Stop when all beam hypotheses reach the EOS token
    )

    # Decode outputs
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def full_RAG(db, lm_name, prompt_template, question):
    lm, tokenizer = load_lm_and_tokenizer(lm_name)
    docs = db.query(question)
    context = format_docs(docs)
    return rag_chain(lm, tokenizer, prompt_template, context, question)

def RAG_with_context(lm_name, prompt_template, question, context):
    lm, tokenizer = load_lm_and_tokenizer(lm_name)
    return rag_chain(lm, tokenizer, prompt_template, context, question)


from langchain_community.llms import HuggingFaceHub


def huggingfacehub_LM_invoke(lm_name, prompt_text):
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

if __name__ == '__main__':
    create_chroma_db()