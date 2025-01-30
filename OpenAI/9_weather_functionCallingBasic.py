import configparser
import os
from openai import OpenAI
import requests
import json
from event_handler import EventHandler  # Import the EventHandler class

# Determine the absolute path to the config file
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, 'config.properties')

# Read the API key from the property file
config = configparser.ConfigParser()
config.read(config_path)
api_key = config.get('DEFAULT', 'api_key')
weather_api_key = config.get('DEFAULT', 'weather_api_key')

client = OpenAI(
  api_key=api_key
)

weatherFunctionSpec = {
    "name": "weather",
    "description": "Get the current weather for a city",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The city"
            }
        },
        "required": ["city"]
    }
}

def get_weather(city):
    base_url = "http://api.weatherapi.com/v1/current.json"
    params = {
        "key": weather_api_key,
        "q": city,
        "aqi": "no"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": "Unable to fetch weather data"}
# Example usage
# city = "Stockholm"
# weather_data = get_weather(city)
# print(weather_data)

def create_chat_completion(messages):
    return client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo", 
        functions=[weatherFunctionSpec]
    )

messages = [
  {"role": "system", "content": "You give short answers to my questions."},
  {"role": "user", "content": "Is it raining in singapore? Whats the temperature?"}
] 

print("--------------------- FIRST -----------------------")
print(messages)


response = create_chat_completion(messages)

responseMessages = response.choices[0].message
print("Got response: ", responseMessages)
messages.append(responseMessages)

# print(response.choices[0].message.content)
# print(response)
#print(response.choices[0].message.function_call)

#FunctionCall(arguments='{"city":"Stockholm"}', name='weather')
if responseMessages.function_call:
    function_call = response.choices[0].message.function_call
    if function_call.name == 'weather':
        arguments = function_call.arguments
        arguments_dict = json.loads(arguments)
        city = arguments_dict.get('city')
        #print(city)  # This should print 'Stockholm'
        weather= get_weather(city)
        print(weather)  # This should print 'Stockholm'


        #call GPT and give it the weather data
        messages.append({"role": "function", "name":"weather", "content": json.dumps(weather)})
        print("--------------------- SECOND -----------------------")
        print(messages)
        response = create_chat_completion(messages)

        responseMessages = response.choices[0].message
        print("Got response: ", responseMessages)

        
        print("--------------------- OUTPUT -----------------------")
        print("Got response: ", responseMessages.content)