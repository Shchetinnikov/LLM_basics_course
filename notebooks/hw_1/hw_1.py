import os
import getpass
from typing import List,Union


from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage, SystemMessage

from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import ServiceContext
from llama_index.embeddings import LangchainEmbedding


# # 1. GigaChat
# Define GigaChat throw langchain.chat_models
def get_giga(giga_key: str) -> GigaChat:
    giga = GigaChat(credentials=giga_key, model="GigaChat", timeout=30, verify_ssl_certs=False)
    giga.verbose = False
    return giga

def test_giga():
    print(test_giga.__name__, 'is running...')
    try:
        giga_key = getpass.getpass("Enter your GigaChat credentials: ")
        giga = get_giga(giga_key)
        print(test_giga.__name__, 'status is OK')
    except:
        print('Error: ', test_giga.__name__, 'is fallen')


# # 2. Prompting
# ### 2.1 Define classic prompt
# Implement a function to build a classic prompt (with System and User parts)
def get_prompt(user_content: str) -> List[Union[SystemMessage, HumanMessage]]:
    messages = [
        SystemMessage(content="You are an assistant who helps people answer their questions."),
        HumanMessage(content=user_content)
    ]
    return messages

# Let's check how it works
def test_prompt():
    print(test_prompt.__name__, 'is running...')
    try:
        giga_key = getpass.getpass("Enter your GigaChat credentials: ")
        giga = get_giga(giga_key)
        user_content = 'Hello!'
        prompt = get_prompt(user_content)
        res = giga(prompt)
        print (res.content)
        print(test_prompt.__name__, 'status is OK')
    except:
        print('Error: ', test_prompt.__name__, 'is fallen')


# ### 3. Define few-shot prompting
# Implement a function to build a few-shot prompt to count even digits in the given number. 
# The answer should be in the format 'Answer: The number {number} consist of {text} even digits.', 
# for example 'Answer: The number 11223344 consist of four even digits.'
def get_prompt_few_shot(number: str) -> List[HumanMessage]:
    few_shot_prompt = f"""
        Example 1: 
        Сount even digits in the given number: 1234
        - 1 % 2 = 1, this is an odd number, number of even digits 0
        - 2 % 2 = 0, this is an even number, number of even digits 1
        - 3 % 2 = 1, this is an odd number, number of even digits 1
        - 4 % 2 = 0, this is an even number, number of even digits 2
        Answer: The number 1234 consist of two even digits.

        Example 2:
        Сount even digits in the given number: 03598624608678942
        - 0 % 2 = 0, this is an even number, number of even digits 1
        - 3 % 2 = 1, this is an odd number, number of even digits 1
        - 5 % 2 = 1, this is an odd number, number of even digits 1
        - 9 % 2 = 1, this is an odd number, number of even digits 1
        - 8 % 2 = 0, this is an even number, number of even digits 2
        - 6 % 2 = 0, this is an even number, number of even digits 3
        - 2 % 2 = 0, this is an even number, number of even digits 4
        - 4 % 2 = 0, this is an even number, number of even digits 5
        - 6 % 2 = 0, this is an even number, number of even digits 6
        - 0 % 2 = 0, this is an even number, number of even digits 7
        - 8 % 2 = 0, this is an even number, number of even digits 8
        - 6 % 2 = 0, this is an even number, number of even digits 9
        - 7 % 2 = 1, this is an odd number, number of even digits 9
        - 8 % 2 = 0, this is an even number, number of even digits 10
        - 9 % 2 = 1, this is an odd number, number of even digits 10 
        - 4 % 2 = 0, this is an even number, number of even digits 11
        - 2 % 2 = 0, this is an even number, number of even digits 12
        Answer:  The number 03598624608678942 consist of twelve even digits.
        
        Сount even digits in the given number: {number}
    """
    message = [
        HumanMessage(content=few_shot_prompt)
    ]
    return message

# Let's check how it works
def test_few_shot():
    print(test_few_shot.__name__, 'is running...')
    try:
        giga_key = getpass.getpass("Enter your GigaChat credentials: ")
        giga = get_giga(giga_key)
        number = '62388712774'
        prompt = get_prompt_few_shot(number)
        res = giga.invoke(prompt)
        print (res.content)
        print(test_few_shot.__name__, 'status is OK')
    except:
        print('Error: ', test_few_shot.__name__, 'is fallen')


# # 4. Llama_index
# Implement your own class to use llama_index. You need to implement some code to build llama_index across your own documents.
# For this task you should use GigaChat Pro.
class LlamaIndex:
    def __init__(self, path_to_data: str, llm: GigaChat):
        self.system_prompt="""
            You are a Q&A assistant. Your goal is to answer questions as
            accurately as possible based on the instructions and context provided.
        """
        self.documents = SimpleDirectoryReader(path_to_data).load_data()
        self.embed_model=LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
        self.service_context = ServiceContext.from_defaults(
                                            chunk_size=1024,
                                            llm=llm,
                                            embed_model=self.embed_model
                                            )
        self.index = VectorStoreIndex.from_documents(self.documents, service_context=self.service_context)
        self.query_engine = self.index.as_query_engine()
        
    def query(self, user_prompt: str) -> str:
        user_input = self.system_prompt + user_prompt
        response = self.query_engine.query(user_input)
        return response.response

# Let's check
def test_llama_index():
    print(test_llama_index.__name__, 'is running...')
    try:
        giga_key = getpass.getpass("Enter your GigaChat credentials: ")
        giga_pro = GigaChat(credentials=giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)

        llama_index = LlamaIndex("./data/", giga_pro)
        res = llama_index.query('Tell about marxist philosophy')
        print (res)
        print(test_llama_index.__name__, 'status is OK')
    except:
        print('Error: ', test_llama_index.__name__, 'is fallen')

    

if __name__ == "__main__":
    os.chdir('./notebooks/hw_1')
    test_giga()
    test_prompt()
    test_few_shot()
    test_llama_index()
    

