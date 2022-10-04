import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import keyboard as key
import random

fig = plt.figure()
ims = []
 
for i in range(100):
    rand = np.random.randn(100) # 100個の乱数を作成
    
    img = plt.plot(rand) # グラフを作成
    plt.title("sample animation")
    plt.ylim(-10,10)
    plt.axis()
 
    ims.append(img) # グラフを配列に追加
 
 
 
# 100枚のプロットを 100ms ごとに表示するアニメーション
ani = animation.ArtistAnimation(fig, ims, interval=100)
plt.show()