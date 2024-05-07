
import os
from typing import Annotated
from dotenv import load_dotenv # type: ignore
from tavily import TavilyClient # type: ignore

from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, register_function
from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
# 加载 .env 文件
load_dotenv()
config_list = [
    {"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]},
    {"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]},
]

'''
现在我们创建三个代理，
包括 user_proxy 作为用户代理， 
writing_assistant 用于生成解决方案（基于初始请求或批评），
以及 reflection_assistant 用于反映和提供批评。
'''
os.makedirs("coding", exist_ok=True)
# Use DockerCommandLineCodeExecutor if docker is available to run the generated code.
# Using docker is safer than running the generated code directly.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

writing_assistant = AssistantAgent(
    name="writing_assistant",
    system_message="You are an writing assistant tasked to write engaging blogpost. You try generate the best blogpost possible for the user's request. If the user provides critique, respond with a revised version of your previous attempts.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

reflection_assistant = AssistantAgent(
    name="reflection_assistant",
    system_message="Generate critique and recommendations on the writing. Provide detailed recommendations, including requests for length, depth, style, etc..",
    llm_config={"config_list": config_list, "cache_seed": None},
)


"""
使用 reflection_assistant 作为反射的嵌套代理构建代理聊天
"""

def reflection_message(recipient, messages, sender, config):
    print("Reflecting...")
    return f"Reflect and provide critique on the following writing. \n\n {recipient.chat_messages_for_summary(sender)[-1]['content']}"


nested_chat_queue = [
    
    {
        "recipient": reflection_assistant,
        "message": reflection_message,
        "max_turns": 1,
    },
]
user_proxy.register_nested_chats(
    nested_chat_queue,
    trigger=writing_assistant,
    # position=4,
)

# Use Cache.disk to cache the generated responses.
# This is useful when the same request to the LLM is made multiple times.
with Cache.disk(cache_seed=42) as cache:
    user_proxy.initiate_chat(
        writing_assistant,
        message="Write an engaging blogpost on the recent updates in AI. "
        "The blogpost should be engaging and understandable for general audience. "
        "Should have more than 3 paragraphes but no longer than 1000 words.",
        max_turns=2,
        cache=cache,
    )