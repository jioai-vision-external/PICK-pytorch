import pandas
import numpy as np
df = pandas.read_csv("/data1/Kanan/PICK-pytorch/data_resized/v4/data_examples_root/combined_shuf.csv", index_col=False)


df = df.iloc[: , 1:]
df.index = np.arange(1, len(df)+1)

print(df.head)


df.to_csv("/data1/Kanan/PICK-pytorch/data_resized/v4/data_examples_root/combined_shuf.csv",header = False)