from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
loader = TextLoader("test.txt")
PDF_data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=5)
all_splits = text_splitter.split_documents(PDF_data)
persist_directory = 'db'
model_name = "/home/lz/Desktop/trans/all-MiniLM-L6-v2"
model_kwargs = {'device': 'cpu'}
embedding = HuggingFaceEmbeddings(
                 model_name=model_name,
                 model_kwargs=model_kwargs)

vectordb = Chroma.from_documents(documents=all_splits, embedding=embedding, persist_directory=persist_directory)


from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

llm = LlamaCpp(
  model_path="/home/lz/Desktop/llama.cpp/testmodels/llama2-7b/ggml-model-f16-q4_0.gguf",
  n_gpu_layers=100,
  n_batch=512,
  n_ctx=2048,
  f16_kv=True,
  callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
  verbose=True,
)

from langchain.chains import LLMChain
from langchain.chains.prompt_selector import ConditionalPromptSelector
from langchain.prompts import PromptTemplate

DEFAULT_LLAMA_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""<<SYS>> 
   You are a helpful assistant eager to assist with providing better Google search results.
   <</SYS>> 

   [INST] Provide an answer to the following question in 150 words. Ensure that the answer is informative, \
       relevant, and concise:
       {question} 
   [/INST]""",
)


DEFAULT_SEARCH_PROMPT = PromptTemplate(
  input_variables=["question"],
  template="""You are a helpful assistant eager to assist with providing better Google search results. \
     Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
     relevant, and concise: \
     {question}""",
)

QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
  default_prompt=DEFAULT_SEARCH_PROMPT,
  conditionals=[(lambda llm: isinstance(llm, LlamaCpp), DEFAULT_LLAMA_SEARCH_PROMPT)],
)

prompt = QUESTION_PROMPT_SELECTOR.get_prompt(llm)
# prompt
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is China known for?"
llm_chain.invoke({"question": question})
retriever = vectordb.as_retriever()

qa = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=retriever,
  verbose=True
)
query = "New China was founded on ?"
qa.invoke(query)