import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime
from sklearn.linear_model import LinearRegression

nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'DataSet/middle_school/TRAIN/fact_data'


def get_station_type(farm_id):
    """判断场站类型 1-5为风电，6-10为光伏"""
    return 'wind' if farm_id <= 5 else 'solar'


def data_preprocess(x_df, y_df):
    x_df = x_df.dropna()
    y_df = y_df.dropna()
    ind = [i for i in y_df.index if i in x_df.index]
    return x_df.loc[ind], y_df.loc[ind]


def train(farm_id):
    x_df = pd.DataFrame()
    station_type = get_station_type(farm_id)
    nwp_train_path = f'DataSet/middle_school/TRAIN/nwp_data_train/{farm_id}'

    for nwp in nwps:
        nwp_path = os.path.join(nwp_train_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        # 根据场站类型选择变量
        if station_type == 'wind':
            # 风电使用 ws, t2m, sp/msl
            target_vars = ['u100', 'v100', 't2m']
            target_vars.append('msl' if nwp == 'NWP_2' else 'sp')
        else:
            # 光伏使用 t2m, tp, tcc, poai, ghi
            target_vars = ['t2m', 'tp', 'tcc', 'poai', 'ghi']

        var_dfs = []
        u, v = None, None

        # 提取目标变量
        for var in nwp_data.channel.values:
            if var not in target_vars:
                continue

            # 提取3x3网格数据
            data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=var
            ).data.values.reshape(-1, 9)  # 自动适配天数

            # 保存风场分量
            if var == 'u100': u = data
            if var == 'v100': v = data

            # 创建特征列
            var_df = pd.DataFrame(
                data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        # 计算风速（仅风电）
        if station_type == 'wind' and u is not None and v is not None:
            ws = np.sqrt(u ** 2 + v ** 2)
            ws_df = pd.DataFrame(
                ws,
                columns=[f"{nwp}_ws_{i}" for i in range(9)]
            )
            var_dfs.append(ws_df)

        x_df = pd.concat([x_df, pd.concat(var_dfs, axis=1)], axis=1)

    # 设置时间索引
    x_df.index = pd.date_range(datetime(1968, 1, 2, 0), datetime(1968, 12, 31, 23), freq='h')

    # 加载标签数据
    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'), index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']

    # 数据预处理
    x_processed, y_processed = data_preprocess(x_df, y_df)
    y_processed[y_processed < 0] = 0

    # 训练模型
    model = LinearRegression()
    model.fit(x_processed, y_processed)
    return model


def predict(model, farm_id):
    x_df = pd.DataFrame()
    station_type = get_station_type(farm_id)
    nwp_test_path = f'DataSet/middle_school/TEST/nwp_data_test/{farm_id}'

    for nwp in nwps:
        nwp_path = os.path.join(nwp_test_path, nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        # 根据场站类型选择变量（与train保持一致）
        if station_type == 'wind':
            target_vars = ['u100', 'v100', 't2m']
            target_vars.append('msl' if nwp == 'NWP_2' else 'sp')
        else:
            target_vars = ['t2m', 'tp', 'tcc', 'poai', 'ghi']

        var_dfs = []
        u, v = None, None

        for var in nwp_data.channel.values:
            if var not in target_vars:
                continue

            data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=var
            ).data.values.reshape(-1, 9)

            if var == 'u100': u = data
            if var == 'v100': v = data

            var_df = pd.DataFrame(
                data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        if station_type == 'wind' and u is not None and v is not None:
            ws = np.sqrt(u ** 2 + v ** 2)
            ws_df = pd.DataFrame(
                ws,
                columns=[f"{nwp}_ws_{i}" for i in range(9)]
            )
            var_dfs.append(ws_df)

        x_df = pd.concat([x_df, pd.concat(var_dfs, axis=1)], axis=1)

    x_df.index = pd.date_range(datetime(1969, 1, 1, 0), datetime(1969, 1, 31, 23), freq='h')
    pred_pw = model.predict(x_df).flatten()

    # 生成15分钟数据
    pred = pd.Series(pred_pw, index=pd.date_range(x_df.index[0], periods=len(pred_pw), freq='h'))
    res = pred.resample('15min').interpolate(method='linear')
    res[res < 0] = 0
    res[res > 1] = 1

    # 新增光伏夜间处理（仅修改此处）
    if station_type == 'solar':
        # 获取小时信息（考虑15分钟精度）
        hours = res.index.hour + res.index.minute / 60
        # 创建时间掩码（20:00-04:00）
        night_mask = (hours >= 20) | (hours < 4)
        res.loc[night_mask] = 0

    return res


# 主程序保持不变
farms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for farm_id in farms:
    model_path = f'models/{farm_id}'
    os.makedirs(model_path, exist_ok=True)

    # 训练模型
    model = train(farm_id)
    with open(os.path.join(model_path, 'baseline_middle_school.pkl'), 'wb') as f:
        pickle.dump(model, f)

    # 生成预测
    pred = predict(model, farm_id)

    # 保存结果
    result_path = 'result/output'
    os.makedirs(result_path, exist_ok=True)
    pred.to_csv(os.path.join(result_path, f'output{farm_id}.csv'))

print('ok')