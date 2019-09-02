import numpy as np
from numpy import sin, cos, sqrt, pi, exp, arctan2
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import seaborn

RE = 6378.137e3
MU = 3.986004418e14

DU = RE
TU = sqrt(DU**3 / MU)
VU = DU / TU
GU = DU / TU**2


# 共面状态方程 state：状态：半径r，速度v，位置角度theta，速度角度gamma，质量m
# t：推力（始终非归一化），phi：推力角度，isp：比冲（s），norm：状态是否采用归一化
# 质量计算始终采用非归一化方式计算，isp也始终采用非归一化方式表示
# isp=None的时候不考虑质量的变化
def coplanar_state_fcn(state, t, phi, isp=None, norm=False):
    r, v, theta, gamma, m = state

    if norm:
        mu = 1
        tm = t / m / GU
    else:
        mu = MU
        tm = t / m

    g = mu / r**2

    if isp is None:
        dot_m = 0
    else:
        dot_m = -t / (GU * isp)

    dot_state = (v * sin(gamma), tm * cos(phi - gamma) - g * sin(gamma),
                 v * cos(gamma) / r,
                 (tm * sin(phi - gamma) - (g - v**2 / r) * cos(gamma)) / v,
                 dot_m)

    return dot_state


class Action:
    def step(self, t, state, interval, norm):
        raise NotImplementedError


# 无推力
class NoneAction(Action):
    def step(self, t, state, interval, norm):
        assert interval > 0

        tf = interval
        t_span = 0, tf
        t_eval = tf,
        result = solve_ivp(
            lambda t, y1: coplanar_state_fcn(y1, 0, 0, None, norm),
            t_span=t_span, y0=state, t_eval=t_eval)

        return (t + interval, tuple(result.y.T[0]))


# 脉冲推力
class ImpulseAction(Action):
    # 初始化参数全为非归一化
    def __init__(self, dv, phi, isp):
        self.dv = dv
        self.phi = phi
        self.isp = isp

    def step(self, t, state, interval, norm):
        assert interval == 0

        r, v, theta, gamma, m = state

        if norm:
            dv = self.dv / VU
        else:
            dv = self.dv

        v = sqrt(v**2 + dv**2 + 2 * v * dv * cos(gamma - self.phi))
        gamma = arctan2(v * sin(gamma) + dv * sin(self.phi),
                        v * cos(gamma) + dv * cos(self.phi))

        if self.isp is not None:
            m *= exp(-self.dv / (GU * self.isp))

        state_ = (r, v, theta, gamma, m)

        return (t, state_)


class CoplanarEnv:
    def __init__(self, state0, norm):
        self.state0 = state0
        self.state = state0
        self.norm = norm
        self.t = 0

    def step(self, action: Action, interval):
        self.t, self.state = action.step(
            self.t, self.state, interval, self.norm)
        return (self.t, self.state)

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass


# 用霍曼转移测试
if __name__ == "__main__":
    # 初始质量
    m0 = 1500

    # 比冲
    isp = 2000

    # 初始和最终轨道半径、速度
    r1, r2 = DU, 1.2 * DU
    v1, v2 = sqrt((MU / r1, MU / r2))

    # 变轨两次加速度
    a = (r1 + r2) / 2
    v1_, v2_ = sqrt(
        (2 * MU * (1 / r1 - 1 / (r1 + r2)), 2 * MU * (1 / r2 - 1 / (r1 + r2))))
    dv1, dv2 = v1_ - v1, v2 - v2_
    impulse_action1 = ImpulseAction(dv1, 0, isp)
    impulse_action2 = ImpulseAction(dv2, 0, isp)
    none_action = NoneAction()

    # 初始轨道时间、变轨轨道时间、最终轨道时间
    t1 = 2 * pi * sqrt(r1**3 / MU)
    dt = pi * sqrt(a**3 / MU)
    t2 = 2 * pi * sqrt(r2**3 / MU)

    t1_norm = t1/TU
    dt_norm = dt/TU
    t2_norm = t2/TU

    # 关键时间点
    t1_mark = t1_norm
    t2_mark = t1_norm + dt_norm
    t3_mark = t1_norm + dt_norm + t2_norm

    # 初始状态向量
    state0 = r1 / DU, v1 / VU, -pi / 2, 0, m0

    # 环境
    env = CoplanarEnv(state0, True)

    # 绘图
    fig = plt.figure()
    ax = fig.gca()

    t = 0
    interval = t2_norm / 1000

    x1, y1 = [], []
    x2, y2 = [], []
    x3, y3 = [], []
    while t < t3_mark:
        # 第一阶段
        if t < t1_mark:
            if t + interval < t1_mark:
                t, each_state = env.step(none_action, interval)
            else:
                _, each_state = env.step(none_action, t1_mark - t)
                t = t1_mark
            x1.append(each_state[0] * cos(each_state[2]))
            y1.append(each_state[0] * sin(each_state[2]))
        # 第一次施加脉冲
        elif t == t1_mark:
            env.step(impulse_action1, 0)
            t, each_state = env.step(none_action, interval)
        # 第二阶段
        elif t < t2_mark:
            if t + interval < t2_mark:
                t, each_state = env.step(none_action, interval)
            else:
                _, each_state = env.step(none_action, t2_mark - t)
                t = t2_mark
            x2.append(each_state[0] * cos(each_state[2]))
            y2.append(each_state[0] * sin(each_state[2]))
        # 第二次施加脉冲
        elif t == t2_mark:
            env.step(impulse_action2, 0)
            t, each_state = env.step(none_action, interval)
        elif t < t3_mark:
            if t + interval < t3_mark:
                t, each_state = env.step(none_action, interval)
            else:
                _, each_state = env.step(none_action, t3_mark - t)
                t = t3_mark
            x3.append(each_state[0] * cos(each_state[2]))
            y3.append(each_state[0] * sin(each_state[2]))

    ax.scatter(x1[0], y1[0], marker='*', color='g')
    ax.plot(x1, y1, color='g')
    ax.scatter(x2[0], y2[0], marker='*', color='r')
    ax.plot(x2, y2, color='r')
    ax.scatter(x3[0], y3[0], marker='*', color='b')
    ax.plot(x3, y3, color='b')

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])

    plt.show()
