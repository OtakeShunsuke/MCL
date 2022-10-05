
import matplotlib.pyplot as plt
from matplotlib import animation
import random
import keyboard

acceleration_x=0
theta=0
x=0
y=0

i=0

while True:
    #前進加速
    if keyboard.read_key() == "w":
        acceleration_x+=0.1
    #左回転加速
    elif keyboard.is_pressed("a"):
        theta+=1
    #右回転加速
    elif keyboard.is_pressed("d"):
        theta-=1
    #ストップ
    elif keyboard.is_pressed("s"):
        acceleration_x=0
        theta=0
    #後退加速
    elif keyboard.is_pressed("x"):
        acceleration_x-=0.1
    #変更なし
    elif keyboard.is_pressed("q"):
        pass

    fig = plt.figure(1,figsize=(8, 8))
    sp = fig.add_subplot(111, aspect='equal')
    sp.set_xlim(-10,10)
    sp.set_ylim(-10,10)

#plt.legend()
    x +=acceleration_x
    y = 0
    print(x)


    sp.scatter(x, y)

    plt.pause(1)
    plt.clf()

    i+=1