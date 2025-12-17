import json
import os
import traceback
from typing import List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from openagents.agents.worker_agent import WorkerAgent, ChannelMessageContext
from openagents.models.event_context import EventContext
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBRegressor

from logger import Logger
from utils import extract_exp_id, NpEncoder

log = Logger(__name__)

project_folder = "/mnt/c/Users/Administrator/PycharmProjects/multi_open_agents"

def apply_features(df: pd.DataFrame, feature_list: List[str],
                      target_col: str = '出力(MW)', time_col: str = '时间'):
    use_station_code, use_time, use_cross, use_wind_cycle, use_norm, use_hour, use_gap \
        = False, False, False, False, False, False, False
    for feature in feature_list:
        if feature == "station_code_feature":
            use_station_code = True
        elif feature == "time_feature":
            use_time = True
        elif feature == "cross_feature":
            use_cross = True
        elif feature == "wind_cycle_feature":
            use_wind_cycle = True
        elif feature == "norm_feature":
            use_norm = True
        elif feature == "hour_feature":
            use_hour = True
        elif feature == "gap_feature":
            use_gap = True

    if use_station_code:
        log.info("**** ID one hot ****")
        df['站点编号_le'] = df['站点编号'].map(lambda x: int(x[1]))
        for col in ['站点编号']:
            unique_value = df[col].unique()
            for value in unique_value:
                df[col + "_" + str(value)] = (df[col] == value)
    if use_time:
        log.info('**** 时间特征 ****')
        df['time'] = pd.to_datetime(df[time_col])
        df['year'] = df['time'].dt.year
        df['day'] = df['time'].dt.day
        df['hour'] = df['time'].dt.hour
        df['hour_split'] = df['time'].dt.hour // 4
        df['month'] = df['time'].dt.month
        df['min'] = df['time'].dt.minute

        log.info("**** 时间独热 ****")
        temp = df['time'].dt.hour // 4
        unique_value = temp.unique()
        for value in unique_value:
            df["hour_" + str(value)] = (temp == value)

        temp = df['time'].dt.month // 4
        unique_value = temp.unique()
        for value in unique_value:
            df["month_" + str(value)] = (temp == value)

        log.info("**** 时间周期化 ****")
        df["hour_sin"] = np.sin(2 * np.pi * df['time'].dt.hour / 24)  # 将 hour 转换为周期值
        df["day_sin"] = np.sin(2 * np.pi * df['time'].dt.day / df['time'].dt.days_in_month)  # 将 day 转换为周期值  ×
    if use_cross:
        log.info('**** 特征交叉 ****')
        # # 相乘、相差、比值
        df['cross'] = df['云量'] * df['相对湿度（%）']
        df['100_minus_10_wind_speed'] = df['100m风速（100m/s）'] - df['10米风速（10m/s）']  #
        df['100_multi_10_wind_speed'] = df['100m风速（100m/s）'] * df['10米风速（10m/s）']  # ×
        df['100_plus_10_wind_speed'] = df['100m风速（100m/s）'] + df['10米风速（10m/s）']  #
        df['100_divide_10_wind_speed'] = df['100m风速（100m/s）'] / df['10米风速（10m/s）']  #
        df['10_divide_100_wind_speed'] = df['10米风速（10m/s）'] / df['100m风速（100m/s）']  #
        log.info("**** 风向特征交叉 ****")
        df['100_minus_10_wind_dir'] = df['100m风向（°)'] - df['10米风向（°)']  # ×
        df['10_minus_100_wind_dir'] = df['10米风向（°)'] - df['100m风向（°)']
        df['100_plus_10_wind_dir'] = df['100m风向（°)'] + df['10米风向（°)']  # ×
        h1 = 100 - 10
        h2 = 200 - 100
        h3 = 215 - 100
        v1 = df['100m风速（100m/s）'] - df['10米风速（10m/s）']
        df['200m风速（200m/s）'] = v1 * (h2 / h1) + df['100m风速（100m/s）']
        df['215m风速（215m/s）'] = v1 * (h3 / h1) + df['100m风速（100m/s）']
        df['200m风速（200m/s）_new'] = v1 * pow(h2 / h1, 0.12) + df['100m风速（100m/s）']
        df['215m风速（215m/s）_new'] = v1 * pow(h3 / h1, 0.12) + df['100m风速（100m/s）']
    if use_wind_cycle:  # 风力周期化
        log.info('**** 风力周期化 ****')
        df['100m风向（°)_sin'] = np.sin(2 * np.pi * df['100m风向（°)'] / 360)  # 将 风向 转换为周期值
        df['10米风向（°)_sin'] = np.sin(2 * np.pi * df['10米风向（°)'] / 360)  # 将 风向 转换为周期值
    if use_norm:
        log.info('**** 特征归一化 ****')
        scaler = MinMaxScaler()
        stand_scaler = StandardScaler()
        df['温度（K）_norm'] = df['温度（K）'] - 273.15  # 没有影响的变换
        df['温度（K）_norm'] = stand_scaler.fit_transform(df['温度（K）_norm'].values.reshape(-1, 1))  # ×
        df['气压(Pa）_minmax_norm'] = scaler.fit_transform(df['气压(Pa）'].values.reshape(-1, 1))
        df['辐照强度（J/m2）_minmax_norm'] = scaler.fit_transform(df['辐照强度（J/m2）'].values.reshape(-1, 1))
        df['降水（m）_norm'] = stand_scaler.fit_transform(df['降水（m）'].values.reshape(-1, 1))  # ×
        df['降水（m）_minmax_norm'] = scaler.fit_transform(df['降水（m）'].values.reshape(-1, 1))  # ×
        df['云量_norm'] = stand_scaler.fit_transform(df['云量'].values.reshape(-1, 1))  # ×
        df['相对湿度（%）_norm'] = stand_scaler.fit_transform(df['相对湿度（%）'].values.reshape(-1, 1))  # ×
    if use_hour:
        df['date_hour'] = df['time'].dt.year.astype(str) + '-' + df['time'].dt.month.astype(str) + '-' + df[
            'time'].dt.day.astype(str) + ':' + df['time'].dt.hour.astype(str)

        for col in tqdm(['气压(Pa）', '相对湿度（%）', '云量', '10米风速（10m/s）', '10米风向（°)', '温度（K）',
                            '辐照强度（J/m2）', '降水（m）', '100m风速（100m/s）', '100m风向（°)']):
            for m in ['mean', 'std', 'skew']:
                df[f'{col}_gby_station_hour_{m}'] = df.groupby(['date_hour', '站点编号'])[col].transform(m)

        del df['date_hour']
    if use_gap:
        log.info("**** 滞后 feature ****")
        log.info("**** GAP feature ****")
        gaps = [-96, -48, -24, -12, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 24, 48, 96]
        for gap in gaps:
            for col in tqdm(['气压(Pa）', '相对湿度（%）', '云量', '10米风速（10m/s）', '10米风向（°)', '温度（K）',
                        '辐照强度（J/m2）', '降水（m）', '100m风速（100m/s）', '100m风向（°)']):
                df[col + f"_shift{gap}"] = df[col].groupby(df['站点编号']).shift(gap)
                df[col + f"_gap{gap}"] = df[col + f"_shift{gap}"] - df[col]
    df_filled = df.fillna(method='ffill')
    df_filled = df_filled.fillna(method='bfill')
    df_filled.loc[:, df.columns.isin([target_col])] = df.loc[:, df.columns.isin([target_col])]
    df = df_filled
    return df

class TrainerAgent(WorkerAgent):
    default_agent_id = "模型训练智能体"

    async def on_direct(self, context: EventContext):
        await self.model_train(context)

    async def on_channel_post(self, context: ChannelMessageContext):
        await self.model_train(context)

    async def model_train(self, context: EventContext):
        log.info(f"模型训练智能体 收到消息：{context.text}")
        ws = self.workspace()
        message_id = context.message_id
        if "模型训练" not in context.text and "训练模型" not in context.text:
            return

        experiment_id = extract_exp_id(context.text)
        round_id = extract_exp_id(context.text, "round")
        experiment_dir = os.path.join(f"{project_folder}/experiments", experiment_id)
        round_str = f"round_{str(round_id).zfill(3)}"
        round_dir = os.path.join(experiment_dir, round_str)
        file_name = "daw_input_data.csv"
        file_path = os.path.join(experiment_dir, file_name)
        df_daw = pd.read_csv(file_path)

        try:
            with open(os.path.join(experiment_dir, "analyze_data_result.json"), "r") as f:
                analyze_data_result = json.load(f)
                log.info(f"读取数据统计结果 {f.name}")
        except Exception as e:
            log.error(f"读取数据统计结果失败: {e}")
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(traceback_str)

        target_col = analyze_data_result["target_col"]
        time_col = analyze_data_result["time_col"]

        # 加载特征配置
        try:
            with open(os.path.join(round_dir, "features.json"), "r") as f:
                feature_list = json.load(f)
                log.info(f"读取特征参数 {f.name}")
        except Exception as e:
            log.error(f"读取特征参数失败: {e}")
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(traceback_str)

        # 加载特征配置
        try:
            with open(os.path.join(round_dir, "features.json"), "r") as f:
                feature_list = json.load(f)
                log.info(f"读取特征参数 {f.name}")
        except Exception as e:
            log.error(f"读取特征参数失败: {e}")
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(traceback_str)

        df_train = apply_features(df_daw, feature_list)

        # 准备模型参数
        try:
            with open(os.path.join(round_dir, "model_params.json"), "r") as f:
                model_config = json.load(f)
                log.info(f"读取特征参数 {f.name}")
        except Exception as e:
            log.error(f"读取特征参数失败: {e}")
            traceback_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
            log.error(traceback_str)

        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 模型开始训练.")

        lgb_config = model_config["lgb"]
        xgb_config = model_config["xgb"]

        estimators = [
            ('lgb', LGBMRegressor(
                n_estimators=lgb_config["n_estimators"],
                learning_rate=lgb_config["learning_rate"],
                random_state=lgb_config["random_state"])),
            ('xgb', XGBRegressor(
                n_estimators=xgb_config["n_estimators"],
                learning_rate=xgb_config["learning_rate"],
                tree_method=xgb_config["tree_method"],
                random_state=xgb_config["random_state"]))
        ]
        model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
        sub_train_df = df_train[df_train['time'] < '2023-02-01 0:0:0']
        sub_val_df = df_train[df_train['time'] >= '2023-02-01 0:0:0']

        feats = [f for f in sub_train_df.columns if f not in [target_col, time_col, 'time', '站点编号', 'min']]
        log.info(f"使用特征:{str(len(feats))}个 \n {feats}")

        model.fit(sub_train_df[feats], sub_train_df[target_col])

        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 模型完成训练.")
        # 保存模型
        joblib.dump(model, round_dir + f"/lgb_xgb.pkl")

        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 模型开始评估.")

        val_pred = model.predict(sub_val_df[feats])

        val_pred[val_pred < 0] = 0

        sub_val_df['pred'] = val_pred
        # 去重基站并计算不重复基站的 MSE
        unique_stations = sub_val_df['站点编号'].unique()
        mse_values = []
        mae_values = []
        r2_values = []
        for station in unique_stations:
            station_data = sub_val_df[sub_val_df['站点编号'] == station]
            col1, col2 = station_data[target_col], station_data['pred']
            mse = mean_squared_error(col1, col2)
            mae = mean_absolute_error(col1, col2)
            r2 = r2_score(col1, col2)
            mse_values.append(mse)
            mae_values.append(mae)
            r2_values.append(r2)

        # 计算 MSE 均值
        mse = sum(mse_values) / len(mse_values)
        mae = sum(mae_values) / len(mae_values)
        r2 = sum(r2_values) / len(r2_values)
        score = 1 / (1 + mse)

        log.info(f"score... {score:.5f} mse...{mse:.5f} mae...{mae:.5f}, 'r2...{r2:.5f}")

        val_result = {
            "score": score,
            "mse": mse,
            "mae": mae,
            "r2": r2
        }
        with open(os.path.join(round_dir, "val_result.json"), "w") as f:
            json.dump(val_result, f, indent=2, ensure_ascii=False, cls=NpEncoder)
            log.info(f"保存验证结果到 {f.name}")


        await ws.channel(context.channel).reply(message_id, f"轮次 {round_id}: 训练完成. 评估指标: \n {val_result}")
        await ws.channel(context.channel).post(f"请模型评估智能体 分析评估结果 实验ID=<exp>{experiment_id}</exp> 轮次ID=<round>{round_id}</round>")


if __name__ == "__main__":
    agent = TrainerAgent()
    agent.start(network_host="localhost", network_port=8700)
    agent.wait_for_stop()
