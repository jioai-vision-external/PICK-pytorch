import imp
import pandas as pd
import tqdm
import glob
import os

gts = pd.read_csv('/data1/datasets/aadhaar_card/v1-real/ground-truths-word-level.csv')


pred_path = "/data1/Kanan/PICK-pytorch/output_one/"
preds = []
for ix, f in enumerate(os.listdir(pred_path)):
    print(f)
    read_path = os.path.join(pred_path, f)
    df = pd.read_csv(read_path, engine='c', encoding='utf-8')
    df.reset_index(inplace=True)
    df["file"] = f.split('.')[0]
    # print(df)
    bb_colors = df["entity"].map(lambda x: "r" if "background" in x else "b").tolist()
    preds.append(df)

preds = pd.concat(preds)
preds.drop("Unnamed: 0", axis=1, inplace=True)

print(preds)

merged = pd.merge(
    gts[["file", "index", "label"]],
    preds[["file", "index", "entity"]],
    on=["file", "index"],
)

print(merged)

from sklearn.metrics import classification_report

print(classification_report(merged["label"], merged["entity"]))