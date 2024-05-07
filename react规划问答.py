# 参考文档：https://microsoft.github.io/autogen/docs/topics/prompting-and-reasoning/react

import os
from typing import Annotated
from dotenv import load_dotenv
from tavily import TavilyClient

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


tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

# NOTE: this ReAct prompt is adapted from Langchain's ReAct agent: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py#L79
ReAct_prompt = """
Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
"""

# Define the ReAct prompt message. Assuming a "question" field is present in the context


def react_prompt_message(sender, recipient, context):
    return ReAct_prompt.format(input=context["question"])

# Setting up code executor.
os.makedirs("coding", exist_ok=True)
# Use docker executor for running code in a container if you have docker installed.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
)

assistant = AssistantAgent(
    name="Assistant",
    system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

# Register the search tool.
register_function(
    search_tool,
    caller=assistant,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

# Cache LLM responses. To get different responses, change the cache_seed value.
with Cache.disk(cache_seed=43) as cache:
    user_proxy.initiate_chat(
        assistant,
        message=react_prompt_message,
        question="btc 2024年的走势?",
        cache=cache,
    )
