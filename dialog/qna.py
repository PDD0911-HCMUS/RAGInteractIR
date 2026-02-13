from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model="llama3.2:3b", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a scene-graph parser. Extract a scene graph as triplets.\n"
     "Return ONLY JSON with the following format:\n"
     "{{\n"
     '  "triplets": [["subject","relation","object"], ...]\n'
     "}}\n"
     "Rules: Use short nouns. Relations should be verbs/prepositions. No extra text."
    ),
    ("human", "Text: {text}")
])

chain = prompt | llm | StrOutputParser()

print(chain.invoke({"text": "A man riding a bicycle next to a car."}))
