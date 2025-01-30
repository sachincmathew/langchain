from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9? Is singapore a country or a city?"),
]


# ---- LangChain OpenAI Chat Model Example ----
# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----
# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")


# ---- LangChain NVIDIA Chat Model Example ----
model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
result = model.invoke(messages)
print(f"Answer from NVIDIA: {result.content}")
