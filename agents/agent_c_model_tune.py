import os
import json

from openagents.config import LLMProviderType
from openagents.models.agent_config import AgentConfig
from openagents.agents.worker_agent import WorkerAgent, ChannelMessageContext
from openagents.models.event_context import EventContext

from logger import Logger
from utils import extract_exp_id, NpEncoder

log = Logger(__name__)

project_folder = "/mnt/c/Users/Administrator/PycharmProjects/multi_open_agents"

class ModelTunerAgent(WorkerAgent):
    default_agent_id = "模型参数调整智能体"

    async def on_direct(self, context: EventContext):
        await self.select_model_params(context)

    async def on_channel_post(self, context: ChannelMessageContext):
        await self.select_model_params(context)

    async def select_model_params(self, context: EventContext):
        log.info(f"模型参数调整智能体 收到消息：{context.text}")
        if "模型参数调整" not in context.text.strip() and "调整模型参数" not in context.text.strip():
            return

        experiment_id = extract_exp_id(context.text)
        round_id = extract_exp_id(context.text, "round")
        exp_dir = os.path.join(f"{project_folder}/experiments", experiment_id)
        round_dir = os.path.join(exp_dir, f"round_{str(round_id).zfill(3)}")

        with open(os.path.join(round_dir, "features.json")) as f:
            features = json.load(f)

        # Default params
        default_params = self.query_model_params()

        instruction = f"""
        你是一位机器学习专家。
        当前已选特征：{features}
        模型默认参数：{default_params}
        请为 LightGBM 和 XGBoost 推荐更优的超参数。
        严格按照下面格式输出一个 JSON 对象，格式和默认参数一致。，你只需要改模型参数里面的值。
        
        {{
          "lgb": {{
            "n_estimators": 255,
            "learning_rate": 0.01,
            "random_state": 42
          }},
          "xgb": {{
            "n_estimators": 255,
            "learning_rate": 0.01,
            "tree_method": "hist",
            "random_state": 42
          }}
        }}
        
        不要使用 Markdown，也不要添加任何解释或额外内容。
        """
        response = await self.run_llm(context, instruction=instruction, disable_mcp=True)
        response = response.actions[0].payload.get("response")
        log.info(f"模型参数调整智能体 输出：{response}")
        # 从response 提取 以"[\"" 开始以 "\"]" 结束的内容
        response_list = response[response.find("{\""): response.rfind("\"}") + 2]
        try:
            model_param = json.loads(response_list.strip())
        except:
            model_param = default_params

        # 把输入的features保存到文件中去
        with open(os.path.join(round_dir, "model_params.json"), "w") as f:
            json.dump(model_param, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            log.info(f"保存模型参数到 {f.name}")

        ws = self.workspace()
        message_id = context.message_id
        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 模型参数已经设置 \n {model_param}.")
        await ws.channel(context.channel).post(f"请模型训练智能体 开始模型训练 实验ID=<exp>{experiment_id}</exp> 轮次ID=<round>{round_id}</round>")

    @staticmethod
    def query_model_params():
        with open(os.path.join(f"{project_folder}/default_params", "model_param.json"), "r") as f:
            model_param = json.load(f)
            log.info(f"查询特征库成功.")
        return model_param


if __name__ == "__main__":
    config = AgentConfig(
        instruction="您是模型调优选择专家.",
        model_name="deepseek-chat",
        provider=LLMProviderType.DEEPSEEK,  # 注意：仍写 "openai"（因协议兼容）
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com"
    )
    agent = ModelTunerAgent(agent_config=config)
    agent.start(network_host="localhost", network_port=8700)
    agent.wait_for_stop()
