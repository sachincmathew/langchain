import configparser
import os
from openai import OpenAI
import json

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

#Conversation to prove that OpenAI does not have real time data
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Is it raining in singapore?"}],
)

print(response.choices[0].message.content)