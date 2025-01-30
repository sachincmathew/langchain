# Prompt Template Docs:
#   https://python.langchain.com/v0.2/docs/concepts/#prompt-templateshttps://python.langchain.com/v0.2/docs/concepts/#prompt-templates

from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# # PART 1: Create a ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}."
prompt_template = ChatPromptTemplate.from_template(template)
print("-----Prompt from Template-----")
prompt = prompt_template.invoke({"topic": "cats"})
print(prompt)
print("--------------1----------------")

# PART 2: Prompt with Multiple Placeholders
template_multiple = " Tell me a {adjective} story about a {animal}."
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})
print("\n----- Prompt with Multiple Placeholders -----\n")
print(prompt)
print("-------------2-----------------")


# PART 3: Prompt with System and Human Messages (Using Tuples)
# ***** This seems to be the best way when we need parameters in bothe System and Human messages.*****
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
print("--------------3----------------")

# # Extra Informoation about Part 3.
# # This does work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers"})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
print("--------------4----------------")

#This does NOT work:
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me {joke_count} jokes."),
]
prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
print(prompt)
print("--------------5----------------")
