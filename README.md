# FarmProject：风电与光伏功率预测

基于多来源数值天气预报（NWP）预测场站归一化发电功率。这里的 farm 指风电/光伏场站，不是农作物分类。

## 机器学习架构

本项目使用表格回归模型，而不是 LSTM、Transformer 或其他深度序列网络。

- `baseline_middle_school.py` 和 `train program 01.py` / `02.py` / `03.py`：线性回归基线与特征处理变体。
- `Train Program Random Forest.py` / `01.py`：随机森林回归。
- `LightGBM.py`、`hahahaha.py`、`Train Program LightGBM.py` / `01.py`、`trainpg.py`、`Train Program Redesigned.py`：LightGBM 梯度提升树回归与特征工程变体。

特征包括天气网格、风速、气温、气压或辐照相关变量；部分版本加入风向和时间特征。预测后插值到 15 分钟，并将归一化功率限制到合法范围。多个脚本是历史实验版本，没有统一指定的“最终最佳模型”。

## 使用

```bash
pip install -r requirements.txt
pip install lightgbm netCDF4 'dask[array]'
python "Train Program LightGBM 01.py"
```

自行准备数据并按脚本约定放到 `DataSet/middle_school/`，包含 TRAIN/TEST 的天气 NetCDF 文件及训练功率标签。场站 1–5 按风电处理、6–10 按光伏处理。不同历史脚本的维度和变量假设不同，换数据后需调整。

## 发布范围

仅发布代码和依赖说明，不包含 `DataSet`、训练模型、预测 CSV、虚拟环境和 IDE 配置。本次进行了 Python 语法检查，没有重新训练、比较版本性能或证明其泛化效果。
