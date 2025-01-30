from langchain_openai import ChatOpenAI

# setx OPENAI_API_KEY KEY
# setx LANGCHAIN_API_KEY KEY

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o-mini")

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9? is singapore a country or city?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
