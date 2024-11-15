import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from threading import Thread

# Load model and tokenizer
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

def chat_response(message, history):
    print(f"Received message: {message}")
    print(f"History: {history}")
    
    messages = []
    for h in history:
        messages.append(h)  # Each h is already a dict with 'role' and 'content'
    messages.append({"role": "user", "content": message})
    
    # Generate response
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    
    # Generate with streaming
    generation_kwargs = dict(
        inputs=inputs,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )
    
    # Create a thread to run the generation
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the response
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        yield partial_message

# Create and launch the Gradio interface
demo = gr.ChatInterface(
    fn=chat_response,
    type="messages",
    title="SmolLM2 Chatbot",
    description="A chatbot powered by SmolLM2-1.7B-Instruct model",
    examples=[
        "What is the capital of France?",
        "How does photosynthesis work?",
        "Write a short poem about autumn."
    ],
    cache_examples=True
)

if __name__ == "__main__":
    demo.launch() 