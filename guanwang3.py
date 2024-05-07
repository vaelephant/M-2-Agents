from autogen import ConversableAgent


local_llm_config={
    "config_list": [
        {
            "model": "NotRequired", # Loaded with LiteLLM command
            "api_key": "NotRequired", # Not needed
            "base_url": "http://0.0.0.0:33144"  # Your LiteLLM URL
        }
    ],
    "cache_seed": None # Turns off caching, useful for testing different models
}


# 创建角色
historian = ConversableAgent(
    "历史学家",
    system_message="You are a historian specializing in 20th-century Asian political history, with a focus on the relationships among the USA, Japan, and China.",
   llm_config=local_llm_config,
    human_input_mode="NEVER"
)

economist = ConversableAgent(
    "经济学家",
    system_message="You are an economist with expertise in international economics, particularly adept at analyzing transnational trade and market trends, especially concerning the USA, Japan, and China.",
   
    llm_config=local_llm_config,
    human_input_mode="NEVER"
)

politician = ConversableAgent(
    "政治家",
    system_message="You are a seasoned politician with extensive experience in international negotiations, well-versed in the political dynamics between the USA, Japan, and China.",
   
    llm_config=local_llm_config,
    human_input_mode="NEVER"
)

common_citizen = ConversableAgent(
    "普通百姓",
    system_message="You are a common citizen who keeps up with international news, particularly interested in the affairs of the USA, Japan, and China, and likes to express personal views and thoughts.",
    
    llm_config=local_llm_config,
    human_input_mode="NEVER"
)



# 模拟讨论
discussion_topic = "如何看待中日美未来的关系?,中文讨论"


try:
    # 初始化对话
    results = historian.initiate_chat(economist, message=discussion_topic, max_turns=2)

    # 检查结果是否可迭代，并打印结果
    if hasattr(results, '__iter__'):
        for result in results:
            print(f"{result.speaker}: {result.message}")
    else:
        print("No iterable results returned or incorrect method implementation.")
except Exception as e:
    print(f"An error occurred during the conversation: {e}")
