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


response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say this is a test"}]
    
    #stream=True,    #not streaming
)

print("Request ID:", response._request_id)
print("x-ratelimit-limit-requests:", response.headers.get('x-ratelimit-limit-requests'))
print("x-ratelimit-limit-tokens:", response.headers.get('x-ratelimit-limit-tokens'))
print("x-ratelimit-remaining-requests:", response.headers.get('x-ratelimit-remaining-requests'))
print("x-ratelimit-remaining-tokens:", response.headers.get('x-ratelimit-remaining-tokens'))
print("x-ratelimit-reset-requests:", response.headers.get('x-ratelimit-reset-requests'))
print("x-ratelimit-reset-tokens:", response.headers.get('x-ratelimit-reset-tokens'))