import pandas as pd, sys, glob, os
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 300)
files = sorted(glob.glob('数据/无人机应急物资运输基础数据/*.xlsx')) + ['结果提交模板.xlsx']
for f in files:
    print("="*100)
    print("FILE:", f)
    xl = pd.ExcelFile(f)
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None)
        print("-"*90)
        print(f"SHEET: {sh}  shape={df.shape}")
        print(df.to_string(max_rows=400, max_cols=60))
