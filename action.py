from scipy.integrate import solve_ivp
from util import MU, DU, TU, VU, GU
import numpy as np
from numpy import sin, cos, sqrt, pi, exp, arctan2


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
                 v * cos(gamma) / r, (tm * sin(phi - gamma) -
                                      (g - v**2 / r) * cos(gamma)) / v, dot_m)

    return dot_state


# 推力基类
class Action:
    def step(self, t, state, tau, norm):
        raise NotImplementedError


# 无推力
class NoneAction(Action):
    def step(self, t, state, tau, norm):
        assert tau > 0

        tf = tau
        t_span = 0, tf
        t_eval = tf,
        result = solve_ivp(
            lambda t, y1_p: coplanar_state_fcn(y1_p, 0, 0, None, norm),
            t_span=t_span,
            y0=state,
            t_eval=t_eval)

        return (t + tau, tuple(result.y.T[0]))


# 脉冲推力
class ImpulseAction(Action):
    # 初始化参数全为非归一化
    def __init__(self, dv, phi, isp):
        self.dv = dv
        self.phi = phi
        self.isp = isp

    def step(self, t, state, tau, norm):
        assert tau == 0

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


# 连续小推力
class LowThrustAction(Action):
    # 初始化参数全为非归一化
    def __init__(self, t, phi, isp):
        self.t = t
        self.phi = phi
        self.isp = isp

    def step(self, t, state, tau, norm):
        assert tau > 0

        tf = tau
        t_span = 0, tf
        t_eval = tf,
        result = solve_ivp(lambda t, y: coplanar_state_fcn(
            y, self.t, self.phi, self.isp, norm),
            t_span=t_span,
            y0=state,
            t_eval=t_eval)

        return (t + tau, tuple(result.y.T[0]))


# 辅助函数，返回对应的action
def action_fcn(genre, args=None):
    if genre == 0:
        return {'genre': 0, 'impulse': None, 'low_thrust': None}
    elif genre == 1:
        return {'genre': 1, 'impulse': np.array(args), 'low_thrust': None}
    elif genre == 2:
        return {'genre': 2, 'impulse': None, 'low_thrust': np.array(args)}
