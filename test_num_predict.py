import time
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0.1,
    num_predict=50
)

prompt_template = PromptTemplate(
    template="""You are Walmart SalesBot. Help the customer find grocery products with prices. Keep response to 1-2 short sentences.
Products in stock:
{product_context}

Customer: {user_input}
Walmart SalesBot:""",
    input_variables=["product_context", "user_input"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

t0 = time.time()
p_context = "Organic Cherry Tomatoes: $2.96, Organic Ketchup: $1.98"
query = "Do you have organic cherry tomatoes and ketchup?"
res = chain.run(product_context=p_context, user_input=query)
t_elapsed = time.time() - t0

print(f"Response in {round(t_elapsed, 2)}s:")
print(res)
