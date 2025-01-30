import configparser
import os
from openai import OpenAI

# Determine the absolute path to the config file
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, 'config.properties')

# Read the API key from the property file
config = configparser.ConfigParser()
config.read(config_path)
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

print(completion.choices[0].message)