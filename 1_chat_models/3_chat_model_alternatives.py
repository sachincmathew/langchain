from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9? Is singapore a country or a city?"),
]


# ---- LangChain OpenAI Chat Model Example ----
# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")
result = model.invoke(messages)
print(f"************************Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")
result = model.invoke(messages)
print(f"************************Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
result = model.invoke(messages)
print(f"************************Answer from Google: {result.content}")


# ---- LangChain NVIDIA Chat Model Example ----
model = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")
result = model.invoke(messages)
print(f"************************Answer from NVIDIA: {result.content}")

# ---- LangChain Groq Chat Model Example ----
model = ChatGroq(model="mixtral-8x7b-32768")
result = model.invoke(messages)
print(f"************************Answer from Groq: {result.content}")
