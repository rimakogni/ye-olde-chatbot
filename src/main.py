
from chatbot import Chatbot


bot = Chatbot()
encoded = bot.encode_prompt("Hello, how are you?")

#print(encoded)
reply = bot.decode_reply([15496, 703, 345, 30]) # Pass in a string of generated token IDs here from your tokenizer
print(reply)