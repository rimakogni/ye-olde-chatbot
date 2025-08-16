from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.system_prompt = "You are a helpful assistant. Respond to the end of this conversation accordingly.\n"
        enc = self.encode_prompt(self.system_prompt)
        self.chat_history_ids = enc["input_ids"]          
        self.chat_history_mask = enc["attention_mask"] 

    #    output example:
    #     {
    #     'input_ids': tensor([[101, 2054, 2003, 1996, 4633, 102]]),
    #     'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
    #     }   
    def encode_prompt(self, prompt: str) -> dict:

        return self.tokenizer(prompt,return_tensors="pt")
   

    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    

    def generate_reply(self, prompt: str) -> str:
        
        new = self.encode_prompt(prompt.rstrip() + "\n")
        new_ids = new["input_ids"]
        new_mask = new["attention_mask"] 

        inputs_ids = torch.cat([self.chat_history_ids, new_ids], dim=1)
        attention_mask = torch.cat([self.chat_history_mask, new_mask], dim=1)
            
        input_len = inputs_ids.shape[1]

        output = self.model.generate(
            input_ids = inputs_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=0.8,            # less random
            top_p=0.8,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.chat_history_ids = output
        self.chat_history_mask = torch.ones_like(output, dtype=torch.long)
        return self.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
    
    def reset_history(self):
        
        enc = self.encode_prompt(self.system_prompt)
        self.chat_history_ids = enc["input_ids"]
        self.chat_history_mask = enc["attention_mask"]