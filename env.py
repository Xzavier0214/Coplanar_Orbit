import numpy as np
from numpy import sin, cos, sqrt, pi, exp, arctan2
import matplotlib.pyplot as plt
import gym
from gym.envs.classic_control import rendering
import space
from util import MU, DU, TU, VU, GU
from action import NoneAction, ImpulseAction
from arrow import make_arrow


# 暂时只有P能进行机动，且初始条件两者都只有相位可指定
class CoplanarEnv(gym.Env):
    def __init__(self,
                 norm=True,
                 theta_p=None,
                 theta_e=None,
                 m_p=1500,
                 m_e=1500,
                 tau=10/TU):

        # 最大半长轴（归一化）
        self.rmax_norm = 1.5

        # 动作集
        self.action_space = space.ActionSpace()

        # 观测集（P轨道半径r，P速度大小v，P位置角度theta，P速度角度gamma，P质量m）,
        #      （E轨道半径r，E速度大小v，E位置角度theta，E速度角度gamma，E质量m）
        observation_box_norm = gym.spaces.Box(
            low=np.array([1, 0, 0, 0, 0]),
            high=np.array([self.rmax_norm, np.inf, 2*pi, 2*pi, np.inf]))
        observation_box = gym.spaces.Box(
            low=np.array([1*DU, 0, 0, 0, 0]),
            high=np.array([self.rmax_norm*DU, np.inf, 2*pi, 0, np.inf]))
        if norm:
            self.observation_space = gym.spaces.Tuple(
                (observation_box_norm, observation_box_norm))
        else:
            self.observation_space = gym.spaces.Tuple(
                (observation_box, observation_box))

        self.norm = norm
        self.m_p = m_p
        self.m_e = m_e
        self.tau = tau

        if theta_p is not None:
            assert 0 <= theta_p <= 2*pi
        else:
            theta_p = 3 * pi / 2

        if theta_e is not None:
            assert 0 <= theta_e <= 2*pi
        else:
            theta_e = 0

        # P、E轨道半径、速度大小
        r_p, r_e = 1, 1.2
        v_p, v_e = sqrt((1/r_p, 1/r_e))

        self.state_p = (r_p, v_p, theta_p, 0, self.m_p)
        self.state_e = (r_e, v_e, theta_e, 0, self.m_e)
        self.state = (self.state_p, self.state_e)

        self.t = 0
        self.viewer = None

        self.trace_p = []
        self.trace_e = []

        self.color_p = (220/255, 20/255, 60/255)
        self.color_e = (60/255, 179/255, 113/255)

        self.last_action_p = None
        self.last_action_e = None

    def step(self, action, tau=None):
        assert self.action_space.contains(action), 'invalid action'

        action_p = action
        self.t, self.state_p = action_p.step(self.t, self.state_p,
                                             self.tau if tau is None else tau,
                                             self.norm)
        self.last_action_p = action_p

        action_e = NoneAction()
        _, self.state_e = action_e.step(self.t, self.state_e,
                                        self.tau if tau is None else tau,
                                        self.norm)
        self.last_action_e = action_e

        self.state = (self.state_p, self.state_e)
        return (self.t, self.state)

    def reset(self):
        pass

    def render(self, trace=True):
        screen_width = 600
        screen_height = 600
        offset = 25
        arrow_size = 50

        width_scale = (screen_width - 2*offset)/(2*self.rmax_norm)
        height_scale = (screen_height - 2*offset)/(2*self.rmax_norm)

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # 绘制P
            circle_p = rendering.make_circle(5)
            self.circle_p_trans = rendering.Transform()
            circle_p.add_attr(self.circle_p_trans)
            circle_p.set_color(*self.color_p)
            self.viewer.add_geom(circle_p)

            # 绘制E
            circle_e = rendering.make_circle(5)
            self.circle_e_trans = rendering.Transform()
            circle_e.add_attr(self.circle_e_trans)
            circle_e.set_color(*self.color_e)
            self.viewer.add_geom(circle_e)

            # 绘制轨迹
            if trace:
                # P轨迹
                line_p = rendering.make_polyline(self.trace_p)
                line_p.set_color(*self.color_p)
                self.viewer.add_geom(line_p)

                # E轨迹
                line_e = rendering.make_polyline(self.trace_e)
                line_e.set_color(*self.color_e)
                self.viewer.add_geom(line_e)

        if self.state is None:
            return None

        r_p, v_p, theta_p, gamma_p, m_p = self.state_p
        r_e, v_e, theta_e, gamma_e, m_e = self.state_e

        x_p = r_p*cos(theta_p)
        y_p = r_p*sin(theta_p)
        x_e = r_e*cos(theta_e)
        y_e = r_e*sin(theta_e)

        if not self.norm:
            x_p /= DU
            y_p /= DU
            x_e /= DU
            y_e /= DU

        circle_p_x = width_scale*x_p + screen_width/2
        circle_e_x = width_scale*x_e + screen_width/2

        circle_p_y = height_scale*y_p + screen_height/2
        circle_e_y = height_scale*y_e + screen_height/2

        self.trace_p.append((circle_p_x, circle_p_y))
        self.trace_e.append((circle_e_x, circle_e_y))

        self.circle_p_trans.set_translation(circle_p_x, circle_p_y)
        self.circle_e_trans.set_translation(circle_e_x, circle_e_y)

        if self.last_action_p is not None:
            arrow_p = self.last_action_p.arrow(arrow_size)
            if arrow_p is not None:
                arrow_p.add_attr(
                    rendering.Transform(translation=(circle_p_x, circle_p_y),
                                        rotation=theta_p))
                arrow_p.set_color(*self.color_p)
                self.viewer.add_geom(arrow_p)

        if self.last_action_e is not None:
            arrow_e = self.last_action_e.arrow(arrow_size)
            if arrow_e is not None:
                arrow_e.add_attr(
                    rendering.Transform(translation=(circle_e_x, circle_e_y),
                                        rotation=theta_e))
                arrow_e.set_color(*self.color_e)
                self.viewer.add_geom(arrow_e)

        return self.viewer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return seed


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
    t1 = pi * sqrt(r1**3 / MU)
    dt = pi * sqrt(a**3 / MU)
    t2 = pi * sqrt(r2**3 / MU)

    t1_norm = t1 / TU
    dt_norm = dt / TU
    t2_norm = t2 / TU

    # 关键时间点
    t1_mark = t1_norm
    t2_mark = t1_norm + dt_norm
    t3_mark = t1_norm + dt_norm + t2_norm

    # 环境
    env = CoplanarEnv()

    # 绘图
    fig = plt.figure()
    ax = fig.gca()

    t = 0
    tau = t2_norm / 100

    x1_p, y1_p, x1_e, y1_e = [], [], [], []
    x2_p, y2_p, x2_e, y2_e = [], [], [], []
    x3_p, y3_p, x3_e, y3_e = [], [], [], []

    x1_p.append(env.state_p[0]*cos(env.state_p[2]))
    y1_p.append(env.state_p[0]*sin(env.state_p[2]))
    x1_e.append(env.state_e[0]*cos(env.state_e[2]))
    y1_e.append(env.state_e[0]*sin(env.state_e[2]))

    while t < t3_mark:
        # env.render()

        # 第一阶段
        if t < t1_mark:
            if t + tau < t1_mark:
                t, (each_state_p, each_state_e) = env.step(none_action, tau)
            else:
                _, (each_state_p, each_state_e) = env.step(
                    none_action, t1_mark - t)
                t = t1_mark
            x1_p.append(each_state_p[0]*cos(each_state_p[2]))
            y1_p.append(each_state_p[0]*sin(each_state_p[2]))
            x1_e.append(each_state_e[0]*cos(each_state_e[2]))
            y1_e.append(each_state_e[0]*sin(each_state_e[2]))
        # 第一次施加脉冲
        elif t == t1_mark:
            _, (each_state_p, each_state_e) = env.step(impulse_action1, 0)
            t = (t1_mark + t2_mark)/2
            x2_p.append(each_state_p[0]*cos(each_state_p[2]))
            y2_p.append(each_state_p[0]*sin(each_state_p[2]))
            x2_e.append(each_state_e[0]*cos(each_state_e[2]))
            y2_e.append(each_state_e[0]*sin(each_state_e[2]))
        # 第二阶段
        elif t < t2_mark:
            if t + tau < t2_mark:
                t, (each_state_p, each_state_e) = env.step(none_action, tau)
            else:
                _, (each_state_p, each_state_e) = env.step(
                    none_action, t2_mark - t)
                t = t2_mark
            x2_p.append(each_state_p[0]*cos(each_state_p[2]))
            y2_p.append(each_state_p[0]*sin(each_state_p[2]))
            x2_e.append(each_state_e[0]*cos(each_state_e[2]))
            y2_e.append(each_state_e[0]*sin(each_state_e[2]))
        # 第二次施加脉冲
        elif t == t2_mark:
            _, (each_state_p, each_state_e) = env.step(impulse_action2, 0)
            t = (t2_mark + t3_mark)/2
            x3_p.append(each_state_p[0]*cos(each_state_p[2]))
            y3_p.append(each_state_p[0]*sin(each_state_p[2]))
            x3_e.append(each_state_e[0]*cos(each_state_e[2]))
            y3_e.append(each_state_e[0]*sin(each_state_e[2]))
        # 第三阶段
        elif t < t3_mark:
            if t + tau < t3_mark:
                t, (each_state_p, each_state_e) = env.step(none_action, tau)
            else:
                _, (each_state_p, each_state_e) = env.step(
                    none_action, t3_mark - t)
                t = t3_mark
            x3_p.append(each_state_p[0]*cos(each_state_p[2]))
            y3_p.append(each_state_p[0]*sin(each_state_p[2]))
            x3_e.append(each_state_e[0]*cos(each_state_e[2]))
            y3_e.append(each_state_e[0]*sin(each_state_e[2]))

    # env.render()

    ax.scatter(x1_p[0], y1_p[0], marker='*', color='g')
    ax.plot(x1_p, y1_p, color='g')
    ax.scatter(x2_p[0], y2_p[0], marker='*', color='r')
    ax.plot(x2_p, y2_p, color='r')
    ax.scatter(x3_p[0], y3_p[0], marker='*', color='b')
    ax.plot(x3_p, y3_p, color='b')

    # ax.plot(x1_e, y1_e, color='g')
    # ax.plot(x2_e, y2_e, color='r')
    # ax.plot(x3_e, y3_e, color='b')

    ax.set_xlim([-1.55, 1.55])
    ax.set_ylim([-1.55, 1.55])

    plt.show()
