import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# 配置参数
nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'DataSet/middle_school/TRAIN/fact_data'

# 定义各气象源的变量
nwp_vars = {
    'NWP_1': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi'],
    'NWP_2': ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi'],
    'NWP_3': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
}


def data_preprocess(x_df, y_df):
    """数据预处理函数"""
    x_df = x_df.dropna()
    y_df = y_df.dropna()
    common_index = y_df.index.intersection(x_df.index)
    return x_df.loc[common_index], y_df.loc[common_index]


def train(farm_id):
    """模型训练函数"""
    # 初始化特征容器
    x_df = pd.DataFrame()

    # 遍历三个气象源
    for nwp in nwps:
        # 加载气象数据
        nwp_path = os.path.join(f'DataSet/middle_school/TRAIN/nwp_data_train/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        # 初始化变量容器
        var_dfs = []
        u, v = None, None

        # 处理每个气象变量
        for var in nwp_vars[nwp]:
            # 提取3x3网格数据（9个点）
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.values.reshape(365 * 24, 9)  # 365天*24小时

            # 保存风场分量
            if var == 'u100': u = var_data
            if var == 'v100': v = var_data

            # 创建特征DataFrame
            var_df = pd.DataFrame(
                var_data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        # 计算风速特征
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(9)])
        var_dfs.append(ws_df)

        # 合并当前气象源特征
        x_df = pd.concat([x_df] + var_dfs, axis=1)

    # 设置时间索引
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0),
                               datetime(1968, 12, 31, 23),
                               freq='h')

    # 加载实际功率数据
    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'),
                       index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']

    # 数据预处理
    x_processed, y_processed = data_preprocess(x_df, y_df)
    y_processed[y_processed < 0] = 0  # 处理负值

    # 修改为随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 使用默认参数
    model.fit(x_processed, y_processed.values.ravel())  # 添加ravel()处理标签形状

    return model


def predict(model, farm_id):
    """预测函数"""
    x_df = pd.DataFrame()

    # 加载测试数据
    for nwp in nwps:
        nwp_path = os.path.join(f'DataSet/middle_school/TEST/nwp_data_test/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        var_dfs = []
        u, v = None, None

        # 处理每个气象变量
        for var in nwp_vars[nwp]:
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.values.reshape(31 * 24, 9)  # 31天*24小时

            if var == 'u100': u = var_data
            if var == 'v100': v = var_data

            var_df = pd.DataFrame(
                var_data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        # 计算风速
        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(9)])
        var_dfs.append(ws_df)

        x_df = pd.concat([x_df] + var_dfs, axis=1)

    # 设置测试时间索引（1969年1月）
    x_df.index = pd.date_range(datetime(1969, 1, 1, 0),
                               datetime(1969, 1, 31, 23),
                               freq='h')

    # 生成预测
    pred_pw = model.predict(x_df).flatten()
    pred = pd.Series(pred_pw,
                     index=pd.date_range(x_df.index[0],
                                         periods=len(pred_pw),
                                         freq='h'))

    # 15分钟插值处理
    res = pred.resample('15min').interpolate(method='linear')
    res[res < 0] = 0  # 下限截断
    res[res > 1] = 1  # 上限截断

    return res


# 主程序流程
if __name__ == "__main__":
    farms = range(1, 11)  # 10个场站

    for farm_id in farms:
        # 创建目录
        model_dir = f'models/{farm_id}'
        os.makedirs(model_dir, exist_ok=True)

        # 训练模型
        print(f"Training model for farm {farm_id}...")
        model = train(farm_id)

        # 保存模型
        with open(os.path.join(model_dir, 'baseline_middle_school.pkl'), 'wb') as f:
            pickle.dump(model, f)

        # 生成预测
        print(f"Generating predictions for farm {farm_id}...")
        pred = predict(model, farm_id)

        # 保存结果
        output_dir = 'result/outputRF'
        os.makedirs(output_dir, exist_ok=True)
        pred.to_csv(os.path.join(output_dir, f'output{farm_id}.csv'))

    print("All tasks completed!")