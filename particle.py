import numpy as np
import random
import math
import matplotlib as plt


class Particle:
    def __init__(self, x, y, theta, w):
        self.weight = w
        self.position = np.array([x, y, theta])


class ParticleFilter:
    def __init__(self, N):
        self.particles = []
        for n in range(N):
            self.particles.append(Particle(0.0,0.0,0.0,1.0/N))
        
        test=[e.weight for e in self.particles]  
        print(test)

    def resampling(self):
        num = len(self.particles)
        weights = [e.w for e in range(num)]
        weights_sum = sum(weights)

        if weights_sum < 1e-100:
            weights = [ w + 1e-100 for w in weights]
        ps = random.choices(self.particles, weights=weights, k=num)    # パーティクルのリストから、weightsのリストの重みに比例した確率で、num個選ぶ
        self.particles = [Particle(*e.pos,1.0/num) for e in ps]          # 選んだリストからパーティクルを取り出し、パーティクルの姿勢から重み1/numの新しいパーティクルを作成
    
    def MoveParticle(self, speed_x, speed_w, motion):
        self.resampling()
        for p in self.particles:
            after=motion(p.position,speed_x,speed_w)
            p.position=after


    # 描画用
    def draw(self,c="blue",lbl="particles"):
        xs = [p.pos[0] for p in self.particles]
        ys = [p.pos[1] for p in self.particles]
        vxs = [math.cos(p.pos[2]) for p in self.particles]
        vys = [math.sin(p.pos[2]) for p in self.particles]
        plt.quiver(xs,ys,vxs,vys,color=c,label=lbl,alpha=0.7)

class Robot:
    def __init__(self,x,y,rad):
        random.seed()
        
        # actual_poses: ロボットの姿勢の真値を1ステップごとに記録したもの
        # （ロボットのクラス内にいるけどロボットはこの情報を使えない）
        self.actual_poses = [np.array([x,y,rad])]
        
        # パーティクルフィルタの準備（パーティクル数30個）
        self.pf = ParticleFilter(30)
    
    #　ロボットの動作をシミュレートするメソッド。シミュレーションだけでなく、ロボットがパーティクルを移動するときにも用いる。
    # つまり実機に実装する場合もこのメソッドが必要となる。雑音の度合いは事前に計測するか、
    # ざっくり決めてフィルタのロバスト性に頼る。
    def motion(self, pos, fw, rot):
        # fwだけ前進してその後rotだけ回転。雑音を混入させる
        actual_fw = random.gauss(fw,fw/10)    #進む距離に対して標準偏差10%の雑音を混入
        dir_error = random.gauss(0.0, math.pi / 180.0 * 3.0) # 前進方向がヨレる雑音を標準偏差3[deg]で混入
        
        px, py, pt = pos
        # 移動後の位置を算出
        x = px + actual_fw * math.cos(pt + dir_error)
        y = py + actual_fw * math.sin(pt + dir_error)
        # 雑音込みの回転各を算出。rotに対して標準偏差10%の雑音を混ぜる
        actual_rot = random.gauss(rot,rot/10)
        t = pt + dir_error + actual_rot      # さらにヨレの分の角度を足す
        
        return np.array([x,y,t])
    
    # ロボットが動くときに呼び出すメソッド。ロボットの位置の更新とパーティクルの位置の更新
    def move(self,fw,rot):
        self.actual_poses.append(self.motion(self.actual_poses[-1],fw,rot))
        self.pf.moveParticles(fw,rot,self.motion)
        
   

        
    # 描画用
    def draw(self,sp,observations):
        for obs in observations:
            for p in self.pf.particles:
                obs.draw(sp,p.pos)
        
        self.pf.draw()
        
        xs = [e[0] for e in self.actual_poses]
        ys = [e[1] for e in self.actual_poses]
        vxs = [math.cos(e[2]) for e in self.actual_poses]
        vys = [math.sin(e[2]) for e in self.actual_poses]
        plt.quiver(xs,ys,vxs,vys,color="red",label="actual robot motion")

def draw(i,observations):
    fig = plt.figure(i,figsize=(8, 8))
    sp = fig.add_subplot(111, aspect='equal')
    sp.set_xlim(-1.0,1.0)
    sp.set_ylim(-0.5,1.5)

    robot.draw(sp,observations)
    
    actual_landmarks.draw()

    plt.legend()

filt=ParticleFilter(10)