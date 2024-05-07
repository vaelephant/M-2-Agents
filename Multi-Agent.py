import os
import autogen
import logging
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from autogen.coding import DockerCommandLineCodeExecutor
import openai  # 导入 OpenAI 模块

# 第一步：设置常量

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Jobname = "请用中文讨论：公司要做一个今日头条上销售袜子的营销方案，ceo负责统筹，coo从市场的角度阐述方案的制定，cto从技术的角度阐述技术实现方案"

# 第二步：设置模型参数

# 加载 .env 文件
load_dotenv()
# openai 的模型
config_list_openai = [{"model": "gpt-4", "api_key": os.environ["OPENAI_API_KEY"]}]


# llama 的模型
# 【重点】使用时候先执行启动命令： litellm --model ollama_chat/llama3
config_list_llama = [
    {
        "model": "NotRequired",  # Loaded with LiteLLM command
        "api_key": "NotRequired",  # Not needed
        "base_url": "http://0.0.0.0:4000"  # Your LiteLLM URL
    }
]

LLM_config = {
    "cache_seed": 42,  # Change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_openai,  # 切换使用模型
    "timeout": 120,
}

#############################测试代码----开始###################################
# # 创建 AssistantAgent 实例
# assistant = AssistantAgent(name="assistant", llm_config={"config_list": config_list_openai})
# # 创建 UserProxyAgent 实例，使用 Docker 进行代码执行
# code_executor = DockerCommandLineCodeExecutor()
# user_proxy = UserProxyAgent(name="user_proxy", code_execution_config={"executor": code_executor})

# user_proxy.initiate_chat(
#     assistant,
#     message="""请用中文讨论：公司要做一个今日头条上销售袜子的营销方案，ceo负责统筹，coo从市场的角度阐述方案的制定，cto从技术的角度阐述技术实现方案""",
# )

# exit()
#############################测试代码---结束###################################


# 第三步  创建 AssistantAgent 实例

initializer = autogen.UserProxyAgent(
    name="Init",
)

ceo = autogen.AssistantAgent(
    name="ceo",
    llm_config=LLM_config,
    system_message="您是首席执行官，负责从战略高度审视私有化大模型对企业长期发展的影响力，评估其对市场竞争力、品牌定位的提升潜力，确保投资回报率与企业愿景的一致性。",
)
cto = autogen.UserProxyAgent(
    name="cto",
    llm_config=LLM_config,
    system_message="您是首席技术官，主导技术架构的选择与搭建，确保私有化大模型的安全性、稳定性和可扩展性，同时考虑技术团队的能力培养与技术支持体系的建立。",
    human_input_mode="NEVER",
)
coo = autogen.AssistantAgent(
    name="coo",
    llm_config=LLM_config,
    system_message="您是首席运营官，聚焦于私有化大模型在优化业务流程、提高运营效率方面的应用，设计实施路径，确保模型的集成能有效减少成本、提升服务质量与客户满意度。",
)

# Define state transition logic
def state_transition(last_speaker, groupchat):
    logging.info(f"Last speaker was: {last_speaker.name}")
    if last_speaker is initializer:
        logging.info("Transitioning to CEO for initial discussion.")
        return ceo
    elif last_speaker is ceo:
        logging.info("CEO has spoken, transitioning to CTO.")
        return cto
    elif last_speaker is cto:
        logging.info("CTO has spoken, transitioning to COO.")
        return coo
    elif last_speaker is coo:
        logging.info("COO has spoken, transitioning back to CEO.")
        return ceo
    return None

# Group chat setup

# 第四步 群聊设置
groupchat = autogen.GroupChat(
    agents=[initializer, ceo, cto, coo],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)

# Group chat manager
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# 第五步 定义群聊函数

# Start discussion function
def start_discussion(Jobname):
    logging.info("Starting the group discussion on the value of privatizing large models.")
    initializer.initiate_chat(manager, message=Jobname)

# Call start_discussion to begin the process
# 第六步 定义群聊函数
start_discussion(Jobname)
