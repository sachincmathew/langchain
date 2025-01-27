import configparser
import os
from openai import OpenAI
import json
from event_handler import EventHandler  # Import the EventHandler class

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

## Create an assistant
# assistant = client.beta.assistants.create(
#     name="Math Tutor",
#     instructions="You are a personal math tutor. Write and run code to answer math questions.",
#     tools=[{"type": "code_interpreter"}],
#     model="gpt-4o-mini"
# )

thread = client.beta.threads.create()

message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
)

# Then, we use the `stream` SDK helper 
# with the `EventHandler` class to create the Run 
# and stream the response.

with client.beta.threads.runs.stream(
  thread_id=thread.id,
  assistant_id="asst_D4KPOBZayIQuxWLRL6dHRksr",#using the id from the openai dashboard
  instructions="Please address the user as Jane Doe. The user has a premium account.",
  event_handler=EventHandler(),
) as stream:
  stream.until_done()