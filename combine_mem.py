import numpy as np
import numpy as np
import os
from tqdm import tqdm
file_list1 = os.listdir(f"./memory")
os.makedirs("./memory_single")

for file1 in file_list1:
    a = None
    file_list2 = os.listdir(f"./memory/{file1}")
    os.makedirs(f"./memory_single/{file1}")
    num = 0
    for file2 in tqdm(file_list2):
        tem = np.load(f"./memory/{file1}/{file2}")
        for i in range(2024):
            np.save(f"./memory_single/{file1}/{num}.npy",tem[i])
            num += 1
