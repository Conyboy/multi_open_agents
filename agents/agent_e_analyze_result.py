import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from openagents.models.agent_config import AgentConfig
from openagents.config.llm_configs import LLMProviderType
from openagents.agents.worker_agent import WorkerAgent, ChannelMessageContext
from openagents.models.event_context import EventContext

from logger import Logger
from utils import extract_exp_id

log = Logger(__name__)

project_folder = "/mnt/c/Users/Administrator/PycharmProjects/multi_open_agents"


class ResultAnalyzerAgent(WorkerAgent):
    default_agent_id = "模型评估智能体"

    async def on_direct(self, context: EventContext):
        await self.analyze_result(context)

    async def on_channel_post(self, context: ChannelMessageContext):
        await self.analyze_result(context)

    async def analyze_result(self, context: EventContext):
        log.info(f"模型评估智能体 收到消息：{context.text}")
        if "分析评估结果" not in context.text and "评估结果分析" not in context.text:
            return

        experiment_id = extract_exp_id(context.text)
        exp_dir = os.path.join(f"{project_folder}/experiments", experiment_id)

        with open(os.path.join(exp_dir, "analyze_data_result.json"), "r") as f:
            analyze_data_result = json.load(f)
            log.info(f"读取数据统计结果 {f.name}")


        file_name = "val_result.json"
        round_dirs = sorted([d for d in os.listdir(exp_dir) if d.startswith("round_")])

        # Load all metrics
        all_metrics = []
        for rd in round_dirs:
            try:
                with open(os.path.join(exp_dir, rd, file_name)) as f:
                    m = json.load(f)
                    m["round"] = rd
                    all_metrics.append(m)
            except:
                continue

        # Plot
        df_metrics = pd.DataFrame(all_metrics)
        plt.figure(figsize=(10, 6))
        plt.plot(df_metrics["round"], df_metrics["mse"], marker="o", label="MSE")
        plt.title("MSE over Rounds")
        plt.xlabel("Round")
        plt.ylabel("MSE")
        plt.legend()
        plot_path = os.path.join(exp_dir, "mse_trend.png")
        plt.savefig(plot_path)
        plt.close()

        # Early stopping logic
        max_rounds = analyze_data_result.get("max_rounds", 5)
        patience = analyze_data_result.get("early_stop_patience", 2)
        if len(round_dirs) >= max_rounds:
            await self.post_to_channel(context.channel, "Max rounds reached. Workflow complete.")
            return

        # Check early stop: if RMSE didn't improve in last `patience` rounds
        if len(df_metrics) >= patience + 1:
            last_rmses = df_metrics["mse"].tail(patience + 1).values
            if all(last_rmses[i] <= last_rmses[i+1] for i in range(len(last_rmses)-1)):
                await self.post_to_channel(context.channel, "Early stopping triggered (no improvement).")
                return

        # Ask LLM whether to continue or adjust strategy
        instruction = f"""
        你是一位 AI 研究协调员。
        当前各轮次的评估结果如下：{all_metrics}
        请判断是否继续下一轮迭代，还是提前终止。
        如果继续，请仅输出："continue"
        如果终止，请输出："stop: [具体原因]"
        不要输出其他任何内容。
        """
        response = await self.run_llm(context, instruction=instruction, disable_mcp=True)
        response = response.actions[0].payload.get("response")
        log.info(f"模型评估智能体 输出：{response}")

        ws = self.workspace()
        message_id = context.message_id
        await ws.channel(context.channel).reply(message_id, f"评估结果分析完成：\n {response}")
        if "continue" in response.lower():
            await ws.channel(context.channel).post(f"请特征选择智能体 开始新一轮的特征选择 实验ID=<exp>{experiment_id}</exp>")
        else:
            await ws.channel(context.channel).post(f"工作流结束: {response}")


if __name__ == "__main__":
    config = AgentConfig(
        instruction="你是一个模型评估专家.",
        model_name="deepseek-chat",
        provider=LLMProviderType.DEEPSEEK,  # 注意：仍写 "openai"（因协议兼容）
        api_key="sk-3d73bcd178b2466cbc23e4e79b32a7f0", # os.getenv("DEEPSEEK_API_KEY"),
        api_base="https://api.deepseek.com"
    )
    agent = ResultAnalyzerAgent(agent_config=config)
    agent.start(network_host="localhost", network_port=8700)
    agent.wait_for_stop()