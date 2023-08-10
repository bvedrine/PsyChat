import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

messages = []

prompt = input("Prompt input: \n")
#'Who are the 10 most prominent 20th century probabilits?'

messages.append({"role": "user", "content": prompt})

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0,
)

chat_response = completion['choices'][0]['message']['content']
print(f'ChatGPT: \n {chat_response}')

