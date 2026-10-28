from pathlib import Path

import pandas as pd

features = [
    "GrLivArea",
    "LotArea",
    "OverallQual",
    "YearBuilt",
    "BedroomAbvGr",
    "FullBath",
    "GarageCars",
    "Neighborhood",
]
columns = features + ["SalePrice"]
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "train.csv"
df = pd.read_csv(DATA_PATH)

# 1. 检查列名
missing_columns = set(columns) - set(df.columns)
if missing_columns:
    raise ValueError(f"缺少字段：{missing_columns}")

print("数据行数、列数：", df.shape)

# 2. 看前 5 套房
print("\n前 5 行：")
print(df[columns].head().to_string(index=False))

# 3. 检查类型和缺失值
print("\n字段类型：")
print(df[columns].dtypes)

print("\n缺失值数量：")
print(df[columns].isna().sum())

# 4. 检查数值范围
print("\n数值统计：")
print(df[columns].describe().to_string())

# 5. 检查类别
print("\n街区类别：")
print(sorted(df["Neighborhood"].dropna().unique()))