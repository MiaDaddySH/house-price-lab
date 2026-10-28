from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 1. 定位项目目录，定义输入字段
# ============================================================

# __file__：当前 Python 文件的路径。
# resolve()：转换成绝对路径。
# parent：当前文件所在的 src 文件夹。
# parent.parent：src 的上一级，也就是 house-price-lab 项目根目录。
#
# 使用脚本位置定位文件后，从项目根目录或 PyCharm 启动都可以。
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
REPORTS_DIR = ROOT / "reports"

# 数值字段：可以进行大小比较或数值计算。
NUMERIC_FEATURES = [
    "GrLivArea",     # 地上居住面积，单位：平方英尺
    "LotArea",       # 地块面积，单位：平方英尺
    "OverallQual",   # 整体材料和装修质量，1～10 级
    "YearBuilt",     # 原始建造年份
    "BedroomAbvGr",  # 地上卧室数量
    "FullBath",      # 地上完整浴室数量
    "GarageCars",    # 车库可容纳的车辆数
]

# 类别字段：街区名称，不能直接当成连续数值使用。
CATEGORICAL_FEATURES = ["Neighborhood"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# 2. 读取数据，分离输入 X 和预测目标 y
# ============================================================

if not DATA_PATH.is_file():
    raise FileNotFoundError(f"找不到训练数据，请检查文件位置：{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# 先检查需要的字段，避免后面因为列名拼错而报错。
required_columns = ["Id", "SalePrice"] + FEATURES
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"CSV 缺少必要字段：{sorted(missing_columns)}")

# X：模型可以看到的房屋信息，是一个包含 8 列的 DataFrame。
# y：模型需要学习预测的真实成交价，是一个 Series。
# SalePrice 不能包含在 FEATURES 中，否则等于把答案交给模型。
X = df[FEATURES]
y = df["SalePrice"]

if y.isna().any() or (y <= 0).any():
    raise ValueError("SalePrice 必须非缺失且大于 0")


# ============================================================
# 3. 划分数据：训练 60%、验证 20%、最终测试 20%
# ============================================================

# 第一次划分：拿出全部数据的 20%，保留作为最终测试集。
# 当前阶段不使用它来选择模型或调整参数。
#
# random_state=42：在输入数据及行顺序不变时，重复运行得到相同划分。
X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

# 第二次划分：从剩余 80% 中拿出 25% 作为验证集。
# 80% × 25% = 全部数据的 20%，因此训练集最终占 60%。
#
# X 和 y 一起划分，保证每套房屋的信息与价格仍然一一对应。
X_train, X_val, y_train, y_val = train_test_split(
    X_dev,
    y_dev,
    test_size=0.25,
    random_state=42,
)


# ============================================================
# 4. 训练中位数基线
# ============================================================

# 基线不使用面积、街区等信息。
# 它只记住 y_train 的中位数，对所有房屋预测相同价格。
# 用途：为真正的机器学习模型提供比较起点。
baseline = DummyRegressor(strategy="median")

# fit：根据训练数据建立模型。
baseline.fit(X_train, y_train)

# predict：对未参与训练的验证集生成预测。
baseline_predictions = baseline.predict(X_val)

# MAE：预测价格与真实价格之差的绝对值，再求平均。
# 单位与 SalePrice 相同，即美元；越小越好。
baseline_mae = mean_absolute_error(y_val, baseline_predictions)


# ============================================================
# 5. 定义数值与类别的预处理流程
# ============================================================

# Pipeline 会按照顺序执行各个步骤。
numeric_pipeline = Pipeline([
    # 用每一列在训练集中的中位数填补该列缺失值。
    ("imputer", SimpleImputer(strategy="median")),

    # 根据训练集的均值和标准差进行标准化。
    # 不同字段的数值尺度相差很大，标准化有利于 Ridge 的正则化。
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    # 用训练集中最常出现的类别填补缺失值。
    ("imputer", SimpleImputer(strategy="most_frequent")),

    # 将街区转换成多个 0/1 字段。
    # 例如：Neighborhood_A、Neighborhood_B……
    ("encoder", OneHotEncoder(
        # 验证或预测时出现训练中没见过的街区，不直接报错。
        # 该字段对应的编码会全部为 0，并不代表模型理解了新街区。
        handle_unknown="ignore",

        # 输出普通数组；当前数据规模较小，方便理解与处理。
        sparse_output=False,
    )),
])

# 按列分配处理方式，再把两部分结果合并。
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])


# ============================================================
# 6. 训练 Ridge 回归并验证
# ============================================================

# 把预处理和模型绑定在一起。
# 这样训练、验证和以后预测都会执行同一套预处理。
ridge = Pipeline([
    ("preprocessor", preprocessor),

    # alpha 控制正则化强度；第一轮固定为 1.0。
    # 正则化会限制回归系数过大。
    ("model", Ridge(alpha=1.0)),
])

# 这一步同时完成：
# 1. 从训练集学习填补值、标准化参数和类别列表。
# 2. 转换训练数据。
# 3. 用转换后的数据训练 Ridge。
#
# 所有这些参数都只从训练集学习，避免验证数据泄漏。
ridge.fit(X_train, y_train)

# predict 会自动使用已经学好的预处理参数。
# 它不会在验证集上重新计算中位数、均值等参数。
ridge_predictions = ridge.predict(X_val)

ridge_mae = mean_absolute_error(y_val, ridge_predictions)


# ============================================================
# 7. 输出结果，比较 Ridge 是否超过基线
# ============================================================

print(f"数据文件：{DATA_PATH}")
print(f"训练集：{len(X_train)} 套")
print(f"验证集：{len(X_val)} 套")
print(f"保留测试集：{len(X_test)} 套")

# :,.2f 表示使用千位分隔符，并保留两位小数。
print(f"\n中位数基线验证 MAE：${baseline_mae:,.2f}")
print(f"Ridge 验证 MAE：${ridge_mae:,.2f}")

if baseline_mae > 0:
    # 正数表示 Ridge 误差更低，负数表示 Ridge 比基线更差。
    improvement = (baseline_mae - ridge_mae) / baseline_mae

    # :.1% 表示以百分比显示，并保留一位小数。
    print(f"Ridge 相对基线的 MAE 降幅：{improvement:.1%}")


# ============================================================
# 8. 保存每套验证房屋的预测结果，方便分析错误
# ============================================================

# 数据划分后仍然保留原始行索引。
# 使用验证集索引，从原始数据中找回对应房屋的 Id 和特征。
# copy() 创建独立表格，便于添加结果列。
details = df.loc[X_val.index, ["Id"] + FEATURES].copy()

# y_val 是 Series，赋值时按照索引对齐。
details["actual_price"] = y_val

# 预测结果是数组，顺序与 X_val 的行顺序一致。
details["baseline_prediction"] = baseline_predictions
details["ridge_prediction"] = ridge_predictions

# 每套房屋的绝对误差，不区分高估或低估。
details["ridge_absolute_error"] = (
    details["actual_price"] - details["ridge_prediction"]
).abs()

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
output_path = REPORTS_DIR / "validation_predictions.csv"

# 把误差最大的房屋排在前面，方便优先检查。
# index=False：不把 pandas 的行索引额外写入 CSV。
details.sort_values(
    "ridge_absolute_error",
    ascending=False,
).to_csv(output_path, index=False)

print(f"\n验证预测明细已保存：{output_path}")