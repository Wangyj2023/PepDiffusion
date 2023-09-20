import numpy as np
import numpy as np
import os
from tqdm import tqdm
file_list1 = os.listdir(f"./memory_c")
os.makedirs("./memory_single_c")

for file1 in file_list1:
    a = None
    file_list2 = os.listdir(f"./memory_c/{file1}")
    os.makedirs(f"./memory_single_c/{file1}")
    num = 0
    for file2 in tqdm(file_list2):
        tem = np.load(f"./memory_c/{file1}/{file2}")
        for i in range(2024):
            np.save(f"./memory_single_c/{file1}/{num}.npy",tem[i])
            num += 1

