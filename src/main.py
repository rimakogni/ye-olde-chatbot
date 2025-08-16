
from chatbot import Chatbot


bot = Chatbot()
# prompt = "What is the weather like today?"
# reply = bot.generate_reply(prompt)
# print(f"Prompt: {prompt}")
# print(f"Reply: {reply}")

prompts = [
    "What's your name?",
    "What do you think about AI?",
    "Sorry, tell me your name again."
]

for prompt in prompts:
    reply = bot.generate_reply(prompt)
    print(f"Prompt: {prompt}")
    print(f"Reply: {reply}\n")