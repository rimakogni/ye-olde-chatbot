
from chatbot import Chatbot


bot = Chatbot()
# prompt = "What is the weather like today?"
# reply = bot.generate_reply(prompt)
# print(f"Prompt: {prompt}")
# print(f"Reply: {reply}")

# prompts = [
#     "What's your name?",
#     "What do you think about AI?",
#     "Sorry, tell me your name again."
# ]

# for prompt in prompts:
#     reply = bot.generate_reply(prompt)
#     print(f"Prompt: {prompt}")
#     print(f"Reply: {reply}\n")

bot = Chatbot()

conv1 = ["Hi!", "What's your name?"]
for p in conv1:
    print("User:", p)
    print("Bot :", bot.generate_reply(p), "\n")

bot.reset_history()  

conv2 = ["New topic: tell me a joke", "One more!"]
for p in conv2:
    print("User:", p)
    print("Bot :", bot.generate_reply(p), "\n")
