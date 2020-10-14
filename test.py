from tqdm.auto import tqdm
import time


for i in tqdm(range(10), desc='loop i', leave=True):
    for j in tqdm(range(5), desc='loop j', leave=False):
        time.sleep(0.1)
    for k in tqdm(range(6), desc='loop k', leave=False):
        time.sleep(0.1)
