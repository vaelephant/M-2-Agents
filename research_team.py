import os  # 导入操作系统模块
from autogen import config_list_from_json  # 从 autogen 模块导入从 JSON 文件中读取配置列表的函数
import autogen  # 导入 autogen 模块

import requests  # 导入请求模块
from bs4 import BeautifulSoup  # 导入解析 HTML 的模块
import json  # 导入处理 JSON 数据的模块

from langchain.agents import initialize_agent  # 从 langchain.agents 模块导入初始化代理的函数
from langchain.chat_models import ChatOpenAI  # 从 langchain.chat_models 模块导入 ChatOpenAI 类
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 从 langchain.text_splitter 模块导入 RecursiveCharacterTextSplitter 类
from langchain.chains.summarize import load_summarize_chain  # 从 langchain.chains.summarize 模块导入 load_summarize_chain 函数
from langchain import PromptTemplate  # 从 langchain 模块导入 PromptTemplate 类
import openai  # 导入 OpenAI 模块
from dotenv import load_dotenv  # 从 dotenv 模块导入 load_dotenv 函数

# 获取各个 API Key
load_dotenv()  # 加载环境变量
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")  # 从 JSON 文件或环境变量中获取配置列表
# 具体可参见第三期视频
openai.api_key = os.getenv("OPENAI_API_KEY")  # 设置 OpenAI API 密钥
serper_api_key = os.getenv("SERPER_API_KEY")  # 设置 Serper API 密钥
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")  # 设置 Browserless API 密钥

# research 工具模块

# 调用 Google search by Serper
def search(query):
    url = "https://google.serper.dev/search"  # 设置 Google 搜索 API 的 URL

    payload = json.dumps({
        "q": query  # 设置搜索的查询字符串
    })
    headers = {
        'X-API-KEY': serper_api_key,  # 设置请求头中的 API 密钥
        'Content-Type': 'application/json'  # 设置请求头中的内容类型
    }

    response = requests.request("POST", url, headers=headers, data=payload)  # 发送 POST 请求

    return response.json()  # 返回响应的 JSON 数据


# 抓取网站内容
def scrape(url: str):
    # 抓取网站内容，并根据目标进行摘要，如果内容太大，则根据目标进行摘要
    # objective 是用户给代理的原始目标和任务，url 是要抓取的网站的 URL

    print("Scraping website...")  # 打印信息
    # 定义请求头
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # 定义要发送的数据
    data = {
        "url": url
    }

    # 将 Python 对象转换为 JSON 字符串
    data_json = json.dumps(data)

    # 发送 POST 请求
    post_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"
    response = requests.post(
        post_url, headers=headers, data=data_json)

    # 检查响应状态码
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")  # 使用 BeautifulSoup 解析响应内容
        text = soup.get_text()  # 获取网页文本内容
        print("CONTENT:", text)  # 打印内容
        if len(text) > 8000:
            output = summary(text)  # 调用 summary 函数
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")  # 打印错误信息


# 总结网站内容
def summary(content):
    llm = ChatOpenAI(temperature=0, model="gpt-4")  # 创建 ChatOpenAI 实例
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)  # 创建 RecursiveCharacterTextSplitter 实例
    docs = text_splitter.create_documents([content])  # 创建文档
    map_prompt = """
    将以下文本写成一个详细的总结，用于研究目的:
    "{text}"
    总结:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text"])  # 创建 PromptTemplate 实例

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',  # 内容切片 防止超过 LLM 的 Token 限制
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )  # 加载 summarize_chain

    output = summary_chain.run(input_documents=docs,)  # 运行 summarize_chain

    return output  # 返回总结内容


# 信息收集
def research(query):
    llm_config_researcher = {
        "functions": [
            {
                "name": "search",
                "description": "google search for relevant information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Google search query",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "scrape",
                "description": "Scraping website content based on url",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Website url to scrape",
                        }
                    },
                    "required": ["url"],
                },
            },
        ],
        "config_list": config_list}

    researcher = autogen.AssistantAgent(
    name="researcher",
    system_message="针对给定的查询进行研究，收集尽可能多的信息，并生成详细的研究结果，包括大量的技术细节和所有相关链接；在研究报告的末尾添加TERMINATE;",
    llm_config=llm_config_researcher,
    ) # 创建研究员助手

    user_proxy = autogen.UserProxyAgent(
        name="User_proxy",
        code_execution_config={"last_n_messages": 2, "work_dir": "research"},
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        function_map={
            "search": search,
            "scrape": scrape,
        }
    )  # 创建用户代理

    user_proxy.initiate_chat(researcher, message=query)  # 启动对话

    # set the receiver to be researcher, and get a summary of the research report
    user_proxy.stop_reply_at_receive(researcher)
    user_proxy.send(
        "请再次给我刚刚生成的研究报告，只返回报告和参考链接。", researcher)

    # return the last message the expert received
    return user_proxy.last_message()["content"]


# 编辑 配置不同的 agent 角色
def write_content(research_material, topic):
    # 创建CEO（首席执行官）助手
    ceo = autogen.AssistantAgent(
        name="CEO",
        system_message="您是首席执行官，负责从战略高度审视私有化大模型对企业长期发展的影响力，评估其对市场竞争力、品牌定位的提升潜力，确保投资回报率与企业愿景的一致性。",
        llm_config={"config_list": config_list},
    )  # 创建CEO助手

    # 创建COO（首席运营官）助手
    coo = autogen.AssistantAgent(
        name="COO",
        system_message="您是首席运营官，聚焦于私有化大模型在优化业务流程、提高运营效率方面的应用，设计实施路径，确保模型的集成能有效减少成本、提升服务质量与客户满意度。",
        llm_config={"config_list": config_list},
    )  # 创建COO助手

    # 创建CTO（首席技术官）助手
    cto = autogen.AssistantAgent(
        name="CTO",
        system_message="您是首席技术官，主导技术架构的选择与搭建，确保私有化大模型的安全性、稳定性和可扩展性，同时考虑技术团队的能力培养与技术支持体系的建立。",
        llm_config={"config_list": config_list},
    )  # 创建CTO助手

    # 创建CHO（首席人力资源官）助手
    cho = autogen.AssistantAgent(
        name="CHO",
        system_message="您是首席人力资源官，评估大模型对人力资源管理的影响，包括人才结构的调整、新技能培训需求的识别以及如何利用模型辅助人才选拔与发展，以适应企业智能化转型的需求。",
        llm_config={"config_list": config_list},
    )  # 创建CHO助手


    writer = autogen.AssistantAgent(
        name="文案撰写员",
        system_message="您是一名专业的作者，特别擅长撰写文章。您的任务是根据编辑的指导和反馈来撰写和优化文章内容。请在完成初稿后根据反馈进行至少两轮内容迭代。在每次迭代后，仔细审阅修改以确保文章内容准确、充实且吸引人。完成后，请在文章末尾添加'TERMINATE'以标识结束。",
        llm_config={"config_list": config_list},
    )  # 创建文案撰写员

    reviewer = autogen.AssistantAgent(
        name="reviewer",
        system_message="""
        您是资深的产品经理，您的主要任务是审阅作者所写的文章，并与公司的高层管理人员（CEO、COO、CTO、CHO）沟通，获取他们对文章内容的反馈。您需要将这些反馈整合后，提供给作者进行内容的迭代修改。请确保进行至少两轮内容迭代后，在对话的末尾添加'TERMINATE'。
        """,
        llm_config={"config_list": config_list},
    )


    user_proxy = autogen.UserProxyAgent(
    name="admin",
    system_message="""
    作为人类管理员，您的角色是监控整个内容创作过程，并在必要时与编辑和作者进行互动。您将不会自动干预流程，除非需要对最终文案进行批准或在特定的决策点需要您的输入。请等待文案结构最终确定后再进行任何干预。
    """,
    code_execution_config=False,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="TERMINATE",  # 终止模式
)


   # 创建组
    groupchat = autogen.GroupChat(
        agents=[user_proxy, ceo, coo, cto, cho, writer, reviewer],
        messages=[],
        max_round=20)
    manager = autogen.GroupChatManager(groupchat=groupchat)

    # 启动对话
    user_proxy.initiate_chat(
        manager, message=f"请开始制定方案 {topic}, 这是相关材料: {research_material}")

    # Writer 负责起草文档
    draft = writer.write("根据提供的材料起草文档。")

    # Reviewer 开始循环获取每位高管的反馈
    satisfied = False  # 初始化满意度标志
    while not satisfied:
        all_feedback_satisfied = True  # 假设所有人满意
        for executive in [ceo, coo, cto, cho]:
            feedback = executive.review("请审阅这份文档草稿，并提供您的反馈。", draft)
            if feedback.lower() != 'satisfied':
                all_feedback_satisfied = False
                draft = writer.revise(f"{executive.name} 的反馈要求进一步修改：{feedback}", draft)
                break  # 一旦有人不满意，跳出循环进行修改

        # 如果所有人都满意，则结束循环
        if all_feedback_satisfied:
            satisfied = True

    # 最终确认
    user_proxy.send(
        "所有高层已确认满意。TERMINATE。", manager)

    # 返回最后接收到的消息作为最终文件
    final_document = user_proxy.last_message()["content"]
    print("最终文件:", final_document)



# 出版
llm_config_content_assistant = {
    "functions": [
        {
            "name": "research",
            "description": "对给定的主题进行研究，返回包括参考链接在内的研究材料。",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "待研究的主题",
                        }
                    },
                "required": ["query"],
            },
        },
        {
            "name": "write_content",
            "description": "根据给定的研究材料和主题撰写内容。",
            "parameters": {
                    "type": "object",
                    "properties": {
                        "research_material": {
                            "type": "string",
                            "description": "特定主题的研究材料，包括可用时的参考链接。",
                        },
                        "topic": {
                            "type": "string",
                            "description": "The topic of the content",
                        }
                    },
                "required": ["research_material", "topic"],
            },
        },
    ],
    "config_list": config_list}

writing_assistant = autogen.AssistantAgent(
    name="writing_assistant",
    system_message="您是一名写作助手，您可以使用研究功能收集关于特定主题的最新信息，然后使用write_content函数撰写非常精心的内容；完成任务后请回复TERMINATE。",
    llm_config=llm_config_content_assistant,
)  # 创建写作助手

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    human_input_mode="TERMINATE",  # 注意此处的模式选择
    function_map={
        "write_content": write_content,  # 调用编辑和信息 组
        "research": research,
    }
)  # 创建用户代理

# 最初 需求 启动干活
user_proxy.initiate_chat(
    writing_assistant, message="本次讨论的主题为：《基于人工智能技术，在未来数据存储市场的产品规划》，每位参与者需深入理解其职能领域，结合行业最佳实践与前沿趋势，提出独到见解与具体行动方案。最终制定出的落地方案应包含详细的技术路线图、预算规划、预期效益分析、风险管理计划及人员培训计划，确保方案的可执行性与可持续发展性。")  # 启动对话
