from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):


        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt,return_tensors="pt")