from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
class Chatbot:
    def __init__(self, model_name="microsoft/DialoGPT-small"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.chat_history_ids = None

    #    output example:
    #     {
    #     'input_ids': tensor([[101, 2054, 2003, 1996, 4633, 102]]),
    #     'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])
    #     }   
    def encode_prompt(self, prompt: str) -> dict:

        return self.tokenizer(prompt,return_tensors="pt")
   
    # input
    # reply_ids = [101, 2009, 2003, 2157, 1998, 2829, 102]
    # skip_special_tokens=True
    #
    # output example:
    #     "It is sunny and warm"
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    

    def generate_reply(self, prompt: str) -> str:
        
        new = self.encode_prompt(prompt.rstrip() + "\n")
        new_ids = new["input_ids"] 

        if self.chat_history_ids is None:
            inputs_ids = new_ids
        else:
            inputs_ids = torch.cat([self.chat_history_ids, new_ids], dim=1)
            
        input_len = inputs_ids.shape[1]
        output = self.model.generate(
            input_ids = inputs_ids,
            do_sample=True,
            temperature=0.9,            # less random
            top_p=0.8,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.chat_history_ids = output
        return self.tokenizer.decode(output[0, input_len:], skip_special_tokens=True).strip()
    
    def reset_history(self):
        
        self.chat_history_ids = None