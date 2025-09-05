import os
import re
import json
import torch
import uvicorn
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from rgbot.ingest import ingest_data

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

vstore = ingest_data(r"C:\Saurabh\Nakul_T4\data\SBI_General_Health_Insurance.pdf")

retriever = vstore.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model='llama3.2:3b')


TEMPLATE = """
You are a helpful and intelligent chatbot.

Respond based on the nature of the question:

- If the question is related to insurance (e.g., health insurance, policies, premiums, claims, etc.), act as an Insurance Assistant. Use the context provided to give a detailed and informative response tailored to the insurance topic, around 400 to 500 words.
- If the question is general (not related to insurance), answer normally like a general-purpose chatbot without referring to the insurance context, keeping responses concise and appropriate unless extra detail is requested.

Context (for insurance-related questions only):
{context}

Question: {question}

Examples:
- Question: Hi
  Answer: Hi how can i help you today.

- Question: What is the premium amount for SBI Super Health Insurance?
  Answer: The premium amount for SBI Super Health Insurance depends on several factors, including the age of the insured, sum insured, and policy term. According to the prospectus...

- Question: How does health insurance work in India?
  Answer: Health insurance in India is a financial product that provides coverage for medical expenses incurred due to illness or injury...

- Question: Who is the Prime Minister of India?
  Answer: As of 2025, the Prime Minister of India is Narendra Modi.

- Question: Can you explain black holes in simple terms?
  Answer: Sure! A black hole is a region in space where gravity is so strong that not even light can escape...

- Question: What are the tax benefits of buying health insurance?
  Answer: Health insurance premiums paid in India qualify for tax deductions under Section 80D of the Income Tax Act...
"""


prompt = ChatPromptTemplate.from_template(TEMPLATE)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return {"error": "No question provided"}
    response = chain.invoke(question)
    return {"answer": response}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9003)