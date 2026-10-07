import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
import lightgbm as lgb
from datetime import datetime

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
    x_df = pd.DataFrame()

    for nwp in nwps:
        nwp_path = os.path.join(f'DataSet/middle_school/TRAIN/nwp_data_train/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        var_dfs = []
        u, v = None, None

        for var in nwp_vars[nwp]:
            # 动态计算reshape的行数
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.values.reshape(-1, 9)  # 修改为动态reshape

            if var == 'u100': u = var_data
            if var == 'v100': v = var_data

            var_df = pd.DataFrame(
                var_data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(9)])
        var_dfs.append(ws_df)

        x_df = pd.concat([x_df] + var_dfs, axis=1)

    x_df.index = pd.date_range(datetime(2024, 1, 1, 0),
                               datetime(2025, 2, 27, 23),
                               freq='15min')

    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'),
                       index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']

    x_processed, y_processed = data_preprocess(x_df, y_df)
    y_processed[y_processed < 0] = 0

    # LightGBM模型配置
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        random_state=42,
        verbosity=-1
    )
    model.fit(x_processed, y_processed.values.ravel())

    return model

def predict(model, farm_id):
    """预测函数"""
    x_df = pd.DataFrame()

    for nwp in nwps:
        nwp_path = os.path.join(f'DataSet/middle_school/TEST/nwp_data_test/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        var_dfs = []
        u, v = None, None

        for var in nwp_vars[nwp]:
            # 动态计算reshape的行数
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.values.reshape(-1, 9)  # 修改为动态reshape

            if var == 'u100': u = var_data
            if var == 'v100': v = var_data

            var_df = pd.DataFrame(
                var_data,
                columns=[f"{nwp}_{var}_{i}" for i in range(9)]
            )
            var_dfs.append(var_df)

        ws = np.sqrt(u ** 2 + v ** 2)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws_{i}" for i in range(9)])
        var_dfs.append(ws_df)

        x_df = pd.concat([x_df] + var_dfs, axis=1)

    x_df.index = pd.date_range(datetime(2025, 3, 1, 0),
                               datetime(2025, 3, 30, 23),
                               freq='15min')

    pred_pw = model.predict(x_df).flatten()
    pred = pd.Series(pred_pw,
                     index=pd.date_range(x_df.index[0],
                                         periods=len(pred_pw),
                                         freq='15min'))

    # 15分钟插值处理
    res = pred.resample('15min').interpolate(method='linear')
    res[res < 0] = 0
    res[res > 1] = 1

    # 光伏场站夜间归零处理
    if 6 <= farm_id <= 10:
        hours = res.index.hour
        night_mask = (hours >= 20) | (hours <= 4)
        res.loc[night_mask] = 0.0

    return res

# 主程序
if __name__ == "__main__":
    farms = range(1, 11)

    for farm_id in farms:
        model_dir = f'models/{farm_id}'
        os.makedirs(model_dir, exist_ok=True)

        print(f"Training model for farm {farm_id}...")
        model = train(farm_id)

        with open(os.path.join(model_dir, 'baseline_middle_school.pkl'), 'wb') as f:
            pickle.dump(model, f)

        print(f"Generating predictions for farm {farm_id}...")
        pred = predict(model, farm_id)

        output_dir = 'result/output'
        os.makedirs(output_dir, exist_ok=True)
        pred.to_csv(os.path.join(output_dir, f'output{farm_id}.csv'))

    print("All tasks completed!")


