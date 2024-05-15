
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

prompt = "You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
            If you don't know the answer, just say that you don't know. Be descriptive and use examples. \
Question: {question} \
Context: {context} \
Answer:"

custom = ChatPromptTemplate.from_template(prompt)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def generate_db(path, filename):
   
    loader = PyPDFLoader(f'{path}')
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, 
                                        persist_directory=f'vectorstores\{filename}')
    retriever = vectorstore.as_retriever()

    chain = generate_chain(custom=custom, retriever=retriever, llm=llm)
    return chain

def fetch_db(docname):
    vectordb = Chroma(persist_directory=f'vectorstores\{docname}', embedding_function=embeddings)
    retriever = vectordb.as_retriever()

    chain = generate_chain(custom=custom, retriever=retriever, llm=llm)
    return chain

def generate_chain(custom, retriever, llm):

    rag_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom
        | llm
        | StrOutputParser()
    )

    rag_chain_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain)

    return rag_chain_source

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

def invoke(rag_chain, question):
    output = rag_chain.invoke(f'{question}')
    ans = output["answer"]
    source = output["context"]
    return ans, source

