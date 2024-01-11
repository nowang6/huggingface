from datasets import load_dataset

ds = load_dataset(path="glue", name="sst2")
print(ds)

ds = load_dataset(path="glue", name="sst2")
t_ds = ds["train"]
# 排序之前
print(t_ds["label"][:10])
# [0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
# 排序之后
sorted = t_ds.sort("label")
print(sorted["label"][:10])
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
print(sorted["label"][-10:])
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# 打乱之前
print(t_ds["label"][:10])
# [0, 0, 1, 0, 0, 0, 1, 1, 0, 1]
# 打乱之后
shuffled = t_ds.shuffle(seed=1)
print(shuffled["label"][:10])
# [1, 1, 1, 1, 0, 1, 1, 1, 0, 0]
