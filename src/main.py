
from chatbot import Chatbot


# bot = Chatbot()
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

# bot = Chatbot()

# conv1 = ["Hi!", "What's your name?"]
# for p in conv1:
#     print("User:", p)
#     print("Bot :", bot.generate_reply(p), "\n")

# bot.reset_history()  

# conv2 = ["New topic: tell me a joke", "One more!"]
# for p in conv2:
#     print("User:", p)
#     print("Bot :", bot.generate_reply(p), "\n")
# main.py


def main():
    bot = Chatbot(model_name='microsoft/DialoGPT-medium')

    print("=== Simple Chat UI ===")
    print("Welcome! You are chatting with your bot.\n")
    print("System prompt:")
    print(bot.system_prompt.strip())
    print("\nType /reset to start a new conversation, /quit to exit.\n")

    while True:
        try:
            user_text = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/quit", "/exit"}:
            print("Bye!")
            break
        if user_text.lower() in {"/reset", "/new"}:
            bot.reset_history()
            print("(history reset)\n")
            continue

        reply = bot.generate_reply(user_text)
        print(f"Bot : {reply}\n")

if __name__ == "__main__":
    main()
