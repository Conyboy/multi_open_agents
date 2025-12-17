import json
import os
import warnings
from datetime import datetime

import pandas as pd
from openagents.agents.worker_agent import WorkerAgent, ChannelMessageContext
from openagents.models.event_context import EventContext

from logger import Logger
from utils import NpEncoder

log = Logger(__name__)

project_folder = "/mnt/c/Users/Administrator/PycharmProjects/multi_open_agents"

class DataAnalyzerAgent(WorkerAgent):
    default_agent_id = "数据加载智能体"

    async def on_direct(self, context: EventContext):
        await self.data_load_and_analyze(context)

    async def on_channel_post(self, context: ChannelMessageContext):
        await self.data_load_and_analyze(context)

    async def data_load_and_analyze(self, context: EventContext):
        log.info(f"数据加载智能体 收到消息：{context.text}")
        msg = context.text.strip()
        if "加载数据" not in msg.lower() and "数据加载" not in msg.lower():
            return

        # 假设用户消息格式：upload target_col=time, date_col=timestamp
        warnings.filterwarnings("ignore")

        data_folder = f"{project_folder}/sample_data"
        target_col = '出力(MW)'
        time_col = '时间'

        train_data = pd.read_csv(f"{data_folder}/A_train_data.csv", encoding='gbk')
        train_info = pd.read_csv(f"{data_folder}/A_train_info.csv", encoding='gbk')
        df_data = train_data.merge(train_info[['站点编号', '装机容量(MW)']], on='站点编号', how='left')
        log.info(df_data.columns)

        df_data = df_data[df_data[target_col] != '<NULL>'].reset_index(drop=True)
        df_data[target_col] = df_data[target_col].astype('float32')

        file_name = "daw_input_data.csv"

        # 基础统计
        file_analysis_results = {
            "filename": file_name,
            "shape": df_data.shape,
            "columns": list(df_data.columns),
            "target_col": target_col,
            "time_col": time_col,
            "target_cor_stat": {},
        }

        for colum in df_data.columns:
            if colum != target_col and colum != time_col and colum != "站点编号":
                file_analysis_results["target_cor_stat"][colum] = df_data[target_col].corr(df_data[colum])

        exp_id = f"experiment_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        experiment_dir = os.path.join(f"{project_folder}/experiments", exp_id)
        # 创建目录
        os.makedirs(experiment_dir, exist_ok=True)
        merged_file_path = os.path.join(experiment_dir, file_name)
        df_data.reset_index().to_csv(merged_file_path, index=False)

        json.dump(file_analysis_results, open(os.path.join(experiment_dir, "analyze_data_result.json"), "w"),
                  indent=2, ensure_ascii=False, cls=NpEncoder)
        ws = self.workspace()
        await ws.channel(context.channel).post("数据分析完成。结果已保存。")
        await ws.channel(context.channel).post("触发特征选择智能体。")
        await ws.channel(context.channel).post(f"请特征选择智能体 开始特征选择 实验ID=<exp>{exp_id}</exp>")


if __name__ == "__main__":
    agent = DataAnalyzerAgent()
    agent.start(network_host="localhost", network_port=8700)
    agent.wait_for_stop()