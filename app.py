# /usr/bin/env python
# coding=utf8


from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex, GPTVectorStoreIndex, PromptHelper, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage

from langchain.chat_models import ChatOpenAI
import gradio as gr
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 文件路径
os.chdir(os.getcwd())

# os.environ["OPENAI_API_KEY"] = ''
def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 2000  
    max_chunk_overlap = 20
    chunk_size_limit = 600
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex.from_documents(documents,service_context=service_context)
    index.storage_context.persist()
    return index


def chatbot(input_text):
    storage_context = StorageContext.from_defaults(persist_dir='./storage')
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    
    print(' request ask : ', input_text)
    response = query_engine.query(input_text)
    print('resp: ', response)
    return response.response


iface = gr.Interface(fn=chatbot,
                     inputs=gr.inputs.Textbox(lines=7, label="输入您的文本"),
                     outputs="text",
                     title="知识库聊天机器人")
# index = construct_index("docs")
iface.launch(share=True)