from src.load_data import load_train
from src.rul_model import add_rul

df = load_train()
df = add_rul(df)

print(df.head())
print("-" * 40)
print("Total rows:", df.shape[0])
print("Total engines:", df["engine_id"].nunique())
print("Max cycle:", df["cycle"].max())
print("Max RUL:", df["RUL"].max())
