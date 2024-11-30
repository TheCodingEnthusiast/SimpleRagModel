import argparse
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import google.generativeai as genai
import os
from dotenv import load_dotenv

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("query_text", type=str, help="The query text.")
args = parser.parse_args()
query_text = args.query_text


model_name = "sentence-transformers/all-mpnet-base-v2"  
embedding_function = HuggingFaceEmbeddings(model_name=model_name)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)


results = db.similarity_search_with_relevance_scores(query_text, k=3)

context_texts = []
for doc, score in results:
    temp=doc.page_content+"\n"+doc.metadata.get("source", "Unknown Source")
    context_texts.append(temp)



context_text = "\n\n---\n\n".join(context_texts)


prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
print(prompt)


genai_api_key = os.environ['GEMINI_API_KEY']
genai.configure(api_key=genai_api_key)
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(prompt)
print(response.text)
