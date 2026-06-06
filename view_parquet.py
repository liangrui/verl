import pandas as pd
import sys
import json

path = sys.argv[1] if len(sys.argv) > 1 else "train.parquet"

df = pd.read_parquet(path)

print(f"文件: {path}")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")
print(f"\n列名: {list(df.columns)}")
print(f"\n数据类型:\n{df.dtypes}")
print(f"\n前3条数据:")

for i, row in df.head(3).iterrows():
    print(f"\n--- 第 {i} 行 ---")
    for col in df.columns:
        val = row[col]
        if isinstance(val, list) or isinstance(val, dict):
            print(f"  {col}: {json.dumps(val, ensure_ascii=False, indent=4)}")
        else:
            print(f"  {col}: {val}")
