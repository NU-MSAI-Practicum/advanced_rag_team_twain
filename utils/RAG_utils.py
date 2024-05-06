import ollama


def gen_text_ollama(sys_msg, user_msg, options=None):
    """
    Generate text using Ollama.

    Args:
        sys_msg (string): The system prompt.
        user_msg (string): The user prompt.
        options (dict): The options for the Ollama model.
    Returns:
        The generated text (string).
    """
    response = ollama.chat(model='llama3', messages=[{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': user_msg}], options=options)
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
    prompt_text = 'tell me aboout your day'
    for np in range(0, 100, 5):
        response = ollama.generate(model='llama3', prompt=prompt_text,
                                   options={'temperature': 0.1, 'seed': 1, 'num_predict': np})
        print(np, len(response['response']))