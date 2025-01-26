import configparser
from openai import OpenAI

# Read the API key from the property file
config = configparser.ConfigParser()
config.read('config.properties')
api_key = config.get('DEFAULT', 'api_key')

client = OpenAI(
  api_key=api_key
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=True,
  messages=[
    {"role": "user", "content": "write a haiku about ai"}
  ]
)

print(completion.choices[0].message);
