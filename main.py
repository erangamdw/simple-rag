from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()  # loads variables from .env

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# response = client.responses.create(
#     model="gpt-4.1-mini",
#     input="hello"
# )

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "who is the captain of the sri lankan cricket team?"}
    ]
)

print(response.output_text)