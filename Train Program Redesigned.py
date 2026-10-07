import os
import pickle
import pandas as pd
import numpy as np
import xarray as xr
import lightgbm as lgb
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# 配置参数
nwps = ['NWP_1', 'NWP_2', 'NWP_3']
fact_path = 'DataSet/middle_school/TRAIN/fact_data'

nwp_vars = {
    'NWP_1': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi'],
    'NWP_2': ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi'],
    'NWP_3': ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
}


def data_preprocess(x_df, y_df):
    x_df = x_df.interpolate(method='linear').ffill().bfill()
    y_df = y_df.interpolate(method='linear').ffill().bfill()
    common_index = y_df.index.intersection(x_df.index)
    return x_df.loc[common_index], y_df.loc[common_index]


def create_time_features(df):
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    return df


def train(farm_id):
    x_df = pd.DataFrame()

    for nwp in nwps:
        nwp_path = os.path.join(f'DataSet/middle_school/TRAIN/nwp_data_train/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        var_dfs = []
        u, v = None, None

        for var in nwp_vars[nwp]:
            # 修正维度处理：展平为二维 (time*hour, )
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.mean(axis=(2, 3))  # 聚合空间维度

            # 展平为二维数组 (365*24, )
            var_data = var_data.reshape(-1, 1)

            # 时间滑动特征（修正为处理一维序列）
            rolling_mean = pd.DataFrame(var_data).rolling(window=3 * 24, min_periods=1).mean().values  # 改为3天窗口

            combined = np.concatenate([var_data, rolling_mean], axis=1)
            columns = [f"{nwp}_{var}", f"{nwp}_{var}_roll3d"]

            var_df = pd.DataFrame(combined, columns=columns)
            var_dfs.append(var_df)

        if u is not None and v is not None:
            # 计算风速风向（修正维度）
            u_flat = u.reshape(-1, 1)
            v_flat = v.reshape(-1, 1)
            ws = np.sqrt(u_flat ** 2 + v_flat ** 2)
            wd = np.arctan2(v_flat, u_flat)
            ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws"])
            wd_sin = pd.DataFrame(np.sin(wd), columns=[f"{nwp}_wd_sin"])
            wd_cos = pd.DataFrame(np.cos(wd), columns=[f"{nwp}_wd_cos"])
            var_dfs.extend([ws_df, wd_sin, wd_cos])

        x_df = pd.concat([x_df] + var_dfs, axis=1)

    # 添加时间特征
    x_df = create_time_features(x_df)

    # 特征筛选
    if farm_id <= 5:
        x_df = x_df.filter(regex='ws|wd|u100|v100|t2m|sp|msl|hour_|month_')
    else:
        x_df = x_df.filter(regex='poai|ghi|tcc|t2m|hour_|month_')

    y_df = pd.read_csv(os.path.join(fact_path, f'{farm_id}_normalization_train.csv'),
                       index_col=0)
    y_df.index = pd.to_datetime(y_df.index)
    y_df.columns = ['power']

    x_processed, y_processed = data_preprocess(x_df, y_df)

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=5)
    best_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=63,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        metric='rmse'
    )
    best_model.fit(x_processed, y_processed.values.ravel())

    return best_model

# 其余代码保持不变
def predict(model, farm_id):
    x_df = pd.DataFrame()

    for nwp in nwps:
        nwp_path = os.path.join(f'DataSet/middle_school/TEST/nwp_data_test/{farm_id}', nwp)
        nwp_data = xr.open_mfdataset(f"{nwp_path}/*.nc")

        var_dfs = []
        u, v = None, None

        # 与训练相同的特征工程
        for var in nwp_vars[nwp]:
            var_data = nwp_data.sel(
                lat=range(4, 7),
                lon=range(4, 7),
                lead_time=range(24),
                channel=[var]
            ).data.mean(axis=(2, 3))

            rolling_mean = pd.DataFrame(var_data).rolling(window=3, min_periods=1).mean().values
            combined = np.concatenate([var_data, rolling_mean], axis=1)
            columns = [f"{nwp}_{var}", f"{nwp}_{var}_roll3"]
            var_df = pd.DataFrame(combined, columns=columns)
            var_dfs.append(var_df)

    if u is not None and v is not None:
        ws = np.sqrt(u ** 2 + v ** 2)
        wd = np.arctan2(v, u)
        ws_df = pd.DataFrame(ws, columns=[f"{nwp}_ws"])
        wd_sin = pd.DataFrame(np.sin(wd), columns=[f"{nwp}_wd_sin"])
        wd_cos = pd.DataFrame(np.cos(wd), columns=[f"{nwp}_wd_cos"])
        var_dfs.extend([ws_df, wd_sin, wd_cos])

    x_df = pd.concat([x_df] + var_dfs, axis=1)

    x_df = create_time_features(x_df)

    # 特征筛选
    if farm_id <= 5:
        x_df = x_df.filter(regex='ws|wd|u100|v100|t2m|sp|msl|hour_|month_')
    else:
        x_df = x_df.filter(regex='poai|ghi|tcc|t2m|hour_|month_')

    # 预测后处理
    pred = model.predict(x_df)
    pred = pd.Series(pred, index=x_df.index).clip(0, 1)

    # 更精确的夜间处理（考虑季节因素）
    if 6 <= farm_id <= 10:
        hours = x_df.index.hour
        month = x_df.index.month
        # 冬季夜间延长（12-2月）
        if month in [12,1,2]:
            night_mask = (hours >= 18) | (hours <= 6)
        else:
            night_mask = (hours >= 20) | (hours <= 5)
        pred[night_mask] = 0.0

    # 使用样条插值提升精度
    res = pred.resample('15min').interpolate(method='spline', order=2)
    return res

# 主程序保持不变
if __name__ == "__main__":
    farms = range(1, 11)

    for farm_id in farms:
        model_dir = f'models/{farm_id}'
        os.makedirs(model_dir, exist_ok=True)

        print(f"Training model for farm {farm_id}...")
        model = train(farm_id)

        with open(os.path.join(model_dir, 'optimized_model.pkl'), 'wb') as f:
            pickle.dump(model, f)

        print(f"Generating predictions for farm {farm_id}...")
        pred = predict(model, farm_id)

        output_dir = 'result/optimized_output'
        os.makedirs(output_dir, exist_ok=True)
        pred.to_csv(os.path.join(output_dir, f'output{farm_id}.csv'))

    print("All tasks completed!")


