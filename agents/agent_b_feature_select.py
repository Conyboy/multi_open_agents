import os
import json

from openagents.config import LLMProviderType
from openagents.models.agent_config import AgentConfig
from openagents.agents.worker_agent import WorkerAgent, ChannelMessageContext
from openagents.models.event_context import EventContext

from logger import Logger
from utils import extract_exp_id

log = Logger(__name__)

project_folder = "/mnt/c/Users/Administrator/PycharmProjects/multi_open_agents"

class FeatureSelectorAgent(WorkerAgent):
    default_agent_id = "特征选择智能体"

    async def on_direct(self, context: EventContext):
        await self.select_features(context)

    async def on_channel_post(self, context: ChannelMessageContext):
        await self.select_features(context)

    async def select_features(self, context: EventContext):
        log.info(f"特征选择智能体收到消息：{context.text}")
        if "特征选择" not in context.text.strip() and "选择特征" not in context.text.strip():
            return

        experiment_id = extract_exp_id(context.text)
        exp_dir = os.path.join(f"{project_folder}/experiments", experiment_id)
        with open(os.path.join(exp_dir, "analyze_data_result.json"), "r") as f:
            analyze_data_result = json.load(f)
            log.info(f"读取数据统计结果 {f.name}")

        feature_dict = self.query_features()

        # Get current round
        round_dirs = [d for d in os.listdir(exp_dir) if d.startswith("round_")]
        round_id = len(round_dirs) + 1
        round_str = f"round_{str(round_id).zfill(3)}"
        round_dir = os.path.join(exp_dir, round_str)
        os.makedirs(round_dir, exist_ok=True)

        # LLM decides which features to use
        instruction = f"""
        你是一位特征工程专家。
        数据已分析完毕：{analyze_data_result}
        可用的特征库如下：{feature_dict}
        请根据目标列的相关性及问题背景，选择合适的特征。
        严格按照下面的格式输出一个 [] 格式的特征名称列表
        
        ["station_code_feature", "time_feature"]
        
        不要包含任何解释、说明或额外文本。
        """
        response = await self.run_llm(context, instruction=instruction, disable_mcp=True)
        response = response.actions[0].payload.get("response")
        log.info(f"特征选择智能体 输出：{response}")
        # 从response 提取 以"[\"" 开始以 "\"]" 结束的内容
        response_list = response[response.find("[\"") : response.rfind("\"]") + 2]
        log.info(f"特征选择智能体 输出：{response}")
        try:
            selected_names = json.loads(response_list.strip())
        except:
            selected_names = list(feature_dict.keys())  # fallback

        selected_features = [f for f in list(feature_dict.keys()) if f in selected_names]

        with open(os.path.join(round_dir, "features.json"), "w") as f:
            json.dump(selected_features, f, indent=2)

        ws = self.workspace()
        message_id = context.message_id
        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 选择了以下特征: {selected_names}")
        await ws.channel(context.channel).post(f"请模型参数调整智能体 开始模型参数调整 实验ID=<exp>{experiment_id}</exp> 轮次ID=<round>{round_id}</round>")

    @staticmethod
    def query_features():
        with open(os.path.join(f"{project_folder}/default_params", "feature_library.json"), "r") as f:
            feature_library = json.load(f)
            log.info(f"查询特征库成功.")
        return feature_library


if __name__ == "__main__":
    config = AgentConfig(
        instruction="您是特征选择专家.",
        model_name="deepseek-chat",
        provider=LLMProviderType.DEEPSEEK,  # 注意：仍写 "openai"（因协议兼容）
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com"
    )
    agent = FeatureSelectorAgent(agent_config=config)
    agent.start(network_host="localhost", network_port=8700)
    agent.wait_for_stop()