"""
Q-Learning 强化学习 — 完整教学实现
====================================

本文件从零实现 Q-Learning 算法，包含：
  1. 自定义网格世界环境 (GridWorld)
  2. 基于 Q-Table 的 Q-Learning agent
  3. 训练 + 实时可视化
  4. 结果分析图表

核心公式:
  Q(s,a) ← Q(s,a) + α [ r + γ·max Q(s',a') - Q(s,a) ]

  α  (alpha) : 学习率     — 新信息覆盖旧信息的程度
  γ  (gamma) : 折扣因子   — 未来奖励的重要性
  ε  (epsilon): 探索率    — 随机探索 vs 利用已知最优

用法:
  python q_learning.py                   # 默认 8×8 网格
  python q_learning.py --size 12         # 12×12 网格
  python q_learning.py --episodes 500   # 500 轮训练
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
import argparse
import time
from collections import deque

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 网格世界环境 ====================

class GridWorld:
    """
    网格世界环境

    智能体从起点出发，目标是到达终点，途中需要避开陷阱。
    每个格子是一个状态，智能体可向上/下/左/右移动。

    状态空间: grid_size × grid_size 个格子
    动作空间: 0=上, 1=下, 2=左, 3=右

    奖励设计:
        - 到达终点: +10
        - 踩到陷阱: -5
        - 每走一步: -0.1（鼓励走最短路径）
    """

    # 动作 → (行偏移, 列偏移)
    ACTIONS = {
        0: (-1, 0),  # 上
        1: (1, 0),   # 下
        2: (0, -1),  # 左
        3: (0, 1),   # 右
    }

    ACTION_NAMES = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    def __init__(self, size=8, n_traps=3, seed=None):
        """
        Args:
            size: 网格边长 (size × size)
            n_traps: 陷阱数量
            seed: 随机种子（可复现）
        """
        self.size = size
        self.n_traps = n_traps
        self.rng = np.random.RandomState(seed)

        # 固定起点和终点位置
        self.start = (size - 1, 0)       # 左下角
        self.goal = (0, size - 1)        # 右上角

        # 随机生成陷阱位置（避开起点和终点）
        self._place_traps()

        # 当前状态
        self.reset()

    def _has_path(self, traps):
        """检查从起点到终点是否存在不经过陷阱的通路 (BFS)"""
        from collections import deque
        visited = set()
        q = deque([self.start])
        visited.add(self.start)
        while q:
            r, c = q.popleft()
            if (r, c) == self.goal:
                return True
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    nxt = (nr, nc)
                    if nxt not in visited and nxt not in traps:
                        visited.add(nxt)
                        q.append(nxt)
        return False

    def _place_traps(self):
        """随机放置陷阱，但确保起点到终点始终存在无陷阱通路"""
        forbidden = {self.start, self.goal}
        candidates = [(r, c) for r in range(self.size) for c in range(self.size)
                      if (r, c) not in forbidden]
        max_attempts = 200
        for _ in range(max_attempts):
            chosen = self.rng.choice(len(candidates),
                                     size=min(self.n_traps, len(candidates)),
                                     replace=False)
            traps = {candidates[i] for i in chosen}
            if self._has_path(traps):
                self.traps = traps
                return
        # 回退：尝试逐个减少陷阱直到通路存在
        for n in range(self.n_traps - 1, -1, -1):
            for _ in range(50):
                chosen = self.rng.choice(len(candidates), size=n, replace=False)
                traps = {candidates[i] for i in chosen}
                if self._has_path(traps):
                    self.traps = traps
                    print(f"  [警告] 陷阱过多堵塞通路，已减少至 {n} 个")
                    return
        self.traps = set()  # 极端情况下不放陷阱

    def reset(self):
        """重置环境到初始状态"""
        self.agent_pos = self.start
        self.done = False
        return self._state_to_idx(self.agent_pos)

    def _state_to_idx(self, pos):
        """将 (row, col) 转换为离散状态编号 0 ~ size²-1"""
        return pos[0] * self.size + pos[1]

    def _idx_to_state(self, idx):
        """将状态编号转回 (row, col)"""
        return (idx // self.size, idx % self.size)

    def step(self, action):
        """
        执行一个动作，返回 (next_state, reward, done, info)

        Args:
            action: 0~3 的整数
        """
        dr, dc = self.ACTIONS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # 撞墙检查
        hit_wall = not (0 <= nr < self.size and 0 <= nc < self.size)

        if hit_wall:
            # 撞墙: 留在原地，给明确的负奖励
            reward = -1.0
            self.done = False
        else:
            self.agent_pos = (nr, nc)
            if self.agent_pos == self.goal:
                reward = 10.0
                self.done = True
            elif self.agent_pos in self.traps:
                reward = -5.0
                self.done = True
            else:
                reward = -0.1  # 步数惩罚
                self.done = False

        return (self._state_to_idx(self.agent_pos), reward, self.done,
                {'pos': self.agent_pos, 'hit_wall': hit_wall})

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return 4

    def render_text(self):
        """在控制台打印当前网格状态"""
        for r in range(self.size):
            row = ''
            for c in range(self.size):
                pos = (r, c)
                if pos == self.agent_pos:
                    row += ' A '
                elif pos == self.goal:
                    row += ' G '
                elif pos in self.traps:
                    row += ' X '
                elif pos == self.start:
                    row += ' S '
                else:
                    row += ' · '
            print(row)
        print()


# ==================== Q-Learning Agent ====================

class QLearningAgent:
    """
    基于 Q-Table 的 Q-Learning Agent

    Q 表是一个 (n_states × n_actions) 的二维数组，存储每个状态-动作对的
    预估累积奖励（Q 值）。

    核心更新规则 (TD 更新):
        Q(s,a) ← Q(s,a) + α [ r + γ·max_a' Q(s',a') - Q(s,a) ]
                              |_________________________|
                                       TD target
                              |__________________________|
                                    TD error (δ)

    动作选择: ε-greedy 策略
        - 以概率 ε 随机探索（均匀随机选动作）
        - 以概率 1-ε 利用已知最优（选 Q 值最大的动作）
        - ε 随时间衰减，初期多探索、后期多利用
    """

    def __init__(self, n_states, n_actions,
                 alpha=0.1,    # 学习率
                 gamma=0.95,   # 折扣因子
                 epsilon=1.0,  # 初始探索率
                 eps_min=0.01, # 最小探索率
                 eps_decay=0.995):  # 每轮衰减系数
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

        # 初始化 Q 表为全零
        self.q_table = np.zeros((n_states, n_actions))

        # 训练统计
        self.episode_rewards = []
        self.episode_steps = []

    def choose_action(self, state):
        """ε-greedy 动作选择"""
        if np.random.random() < self.epsilon:
            # 探索：随机选择动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择 Q 值最大的动作（平局时随机打破）
            q_values = self.q_table[state]
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            return np.random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Q-Learning 核心更新 (off-policy TD 学习)

        关键：使用 max_a' Q(s', a') 而非实际下一步动作。
        这意味着智能体总是朝着"理论上最优"的方向更新，
        即使当前策略选了一个探索动作。
        """
        # 当前 Q(s, a)
        current_q = self.q_table[state, action]

        if done:
            # 终止状态：没有后续状态，TD target = r
            td_target = reward
        else:
            # TD target = r + γ * max_a' Q(s', a')
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # TD error δ = target - current
        td_error = td_target - current_q

        # Q 表更新
        self.q_table[state, action] += self.alpha * td_error

        return td_error

    def decay_epsilon(self):
        """每轮结束后衰减探索率"""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ==================== 训练循环 ====================

def train(env, agent, episodes=300, render_every=None, verbose=True):
    """
    训练 Q-Learning agent

    Args:
        env: GridWorld 环境
        agent: QLearningAgent
        episodes: 训练总轮数
        render_every: 每隔多少轮打印一次（None 则不打印）
        verbose: 是否打印训练进度

    Returns:
        history: dict 包含每一步的 (state, action, reward, td_error, epsilon)
    """
    history = {
        'episode': [], 'step': [],
        'state': [], 'action': [], 'reward': [],
        'td_error': [], 'epsilon': [], 'success': [], 'hit_wall': [],
    }

    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = agent.choose_action(state)

            next_state, reward, done, info = env.step(action)

            # 撞墙也要学习——让 Q(s, 撞墙动作) 学成负值
            td_err = agent.learn(state, action, reward, next_state, done)

            # 记录
            history['episode'].append(ep)
            history['step'].append(steps)
            history['state'].append(state)
            history['action'].append(action)
            history['reward'].append(reward)
            history['td_error'].append(td_err)
            history['epsilon'].append(agent.epsilon)
            history['hit_wall'].append(info.get('hit_wall', False))

            total_reward += reward
            steps += 1
            state = next_state

            # 防止死循环（如果 Q 表学坏了可能导致无限循环）
            if steps > env.size * env.size * 3:
                break

        # 记录本轮结果
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(steps)
        history['success'].append(reward == 10.0)  # 到达终点

        agent.decay_epsilon()

        # 打印
        if render_every and ep % render_every == 0:
            if verbose:
                status = '✅ 成功' if reward == 10.0 else '❌ 失败'
                print(f"Episode {ep:4d}/{episodes} | "
                      f"Reward: {total_reward:6.2f} | "
                      f"Steps: {steps:3d} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"{status}")

    return history


# ==================== 可视化 ====================

def visualize_grid_world(env, agent, ax=None):
    """
    可视化网格世界 + Q 值 + 策略

    绘制内容:
      - 网格底色
      - 起点(S)、终点(G)、陷阱(X)
      - 每个格子的最优动作箭头
      - Q 值热力叠加
      - 当前最优路径（如果有）
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    size = env.size

    # 计算每个状态的 V(s) = max_a Q(s,a) 用于热力图
    v_values = np.max(agent.q_table, axis=1).reshape(size, size)

    # 背景热力图
    im = ax.imshow(v_values, cmap='YlOrRd', origin='upper', alpha=0.6)

    # 绘制网格线
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # 标记特殊格子
    ax.text(env.start[1], env.start[0], 'S', ha='center', va='center',
            fontsize=16, fontweight='bold', color='blue')
    ax.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
            fontsize=16, fontweight='bold', color='green')
    for trap in env.traps:
        ax.text(trap[1], trap[0], 'X', ha='center', va='center',
                fontsize=14, fontweight='bold', color='red')

    # 在每个格子绘制最优动作箭头
    for r in range(size):
        for c in range(size):
            state = r * size + c
            if (r, c) == env.goal or (r, c) in env.traps:
                continue

            best_a = np.argmax(agent.q_table[state])
            q_vals = agent.q_table[state]

            # 如果所有 Q 值都是 0（未访问），跳过
            if np.all(q_vals == 0):
                continue

            dr, dc = GridWorld.ACTIONS[best_a]
            ax.arrow(c, r, dc * 0.3, dr * 0.3,
                     head_width=0.2, head_length=0.15,
                     fc='black', ec='black', alpha=0.7)

    # 在每个格子标注 V 值
    for r in range(size):
        for c in range(size):
            val = v_values[r, c]
            if val != 0:
                ax.text(c, r + 0.25, f'{val:.1f}',
                        ha='center', va='center', fontsize=7, color='darkred')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Grid World + Q-Values + Policy', fontsize=13, fontweight='bold')

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.3, label='S: 起点'),
        Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.3, label='G: 终点'),
        Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.3, label='X: 陷阱'),
        Line2D([0], [0], marker='>', color='black', label='最优动作'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=9)

    return im


def plot_training_curves(agent, history):
    """绘制训练曲线综合面板"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    episodes = range(1, len(agent.episode_rewards) + 1)

    # 平滑函数
    def smooth(data, window=20):
        if len(data) < window:
            return data
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='valid')

    # 1. 每轮奖励
    ax = axes[0, 0]
    ax.plot(episodes, agent.episode_rewards, alpha=0.3, color='steelblue',
            linewidth=0.8, label='原始')
    if len(agent.episode_rewards) >= 20:
        smoothed = smooth(agent.episode_rewards, window=20)
        ax.plot(range(20, len(agent.episode_rewards) + 1), smoothed,
                color='steelblue', linewidth=2, label='平滑 (窗口=20)')
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='最优奖励')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('每轮累积奖励', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. 每轮步数
    ax = axes[0, 1]
    ax.plot(episodes, agent.episode_steps, alpha=0.3, color='coral',
            linewidth=0.8, label='原始')
    if len(agent.episode_steps) >= 20:
        smoothed = smooth(agent.episode_steps, window=20)
        ax.plot(range(20, len(agent.episode_steps) + 1), smoothed,
                color='coral', linewidth=2, label='平滑 (窗口=20)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps')
    ax.set_title('每轮完成步数', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. 探索率衰减
    ax = axes[1, 0]
    epsilons = [history['epsilon'][i] for i in range(len(history['epsilon']))
                if history['step'][i] == 0]  # 每轮起始的 ε
    ax.plot(range(1, len(epsilons) + 1), epsilons, color='purple', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('探索率 ε 衰减曲线', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 4. 成功率滑动窗口
    ax = axes[1, 1]
    successes = np.array(history['success'])
    window = min(50, len(successes) // 4)
    if window > 0 and len(successes) >= window:
        success_rate = np.convolve(successes, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(successes) + 1), success_rate * 100,
                color='green', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'成功率 (滑动窗口={window})', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Q-Learning 训练曲线', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig


def plot_q_table_heatmap(agent, env):
    """展示每个动作的 Q 值热力图"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    action_names = ['上 (0)', '下 (1)', '左 (2)', '右 (3)']

    vmin = agent.q_table.min()
    vmax = agent.q_table.max()

    for a in range(4):
        ax = axes[a]
        q_map = agent.q_table[:, a].reshape(env.size, env.size)
        im = ax.imshow(q_map, cmap='coolwarm', origin='upper',
                        vmin=vmin, vmax=vmax)
        ax.set_title(f'Q(s, {action_names[a]})', fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # 标注值
        for r in range(env.size):
            for c in range(env.size):
                val = q_map[r, c]
                ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                        fontsize=7, color='black' if abs(val) < vmax * 0.5 else 'white')

    plt.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Q-value')
    plt.suptitle('Q-Table: 每个动作的 Q 值分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_optimal_path(env, agent):
    """从起点出发沿最优策略走出的路径"""
    fig, ax = plt.subplots(figsize=(8, 8))
    size = env.size

    # 热力底色
    v_values = np.max(agent.q_table, axis=1).reshape(size, size)
    ax.imshow(v_values, cmap='YlOrRd', origin='upper', alpha=0.4)

    # 网格
    for i in range(size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.3)

    # 标志
    ax.text(env.start[1], env.start[0], 'S', ha='center', va='center',
            fontsize=18, fontweight='bold', color='blue')
    ax.text(env.goal[1], env.goal[0], 'G', ha='center', va='center',
            fontsize=18, fontweight='bold', color='green')
    for trap in env.traps:
        ax.text(trap[1], trap[0], 'X', ha='center', va='center',
                fontsize=16, fontweight='bold', color='red')

    # 贪婪策略走一遍
    state = env._state_to_idx(env.start)
    path = [env.start]
    visited = set()
    for _ in range(size * size * 2):
        if state in visited or (state // size, state % size) == env.goal:
            break
        visited.add(state)
        action = np.argmax(agent.q_table[state])
        dr, dc = GridWorld.ACTIONS[action]
        r, c = state // size, state % size
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            path.append((nr, nc))
            state = nr * size + nc
        else:
            break

    # 画路径
    rows, cols = zip(*path)
    ax.plot(cols, rows, 'b-o', linewidth=2, markersize=6, alpha=0.7,
            label=f'最优路径 ({len(path)} 步)')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('贪婪策略下的最优路径', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    return fig


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='Q-Learning 网格世界 — 教学演示')
    parser.add_argument('--size', type=int, default=8,
                        help='网格边长 (默认: 8)')
    parser.add_argument('--traps', type=int, default=20,
                        help='陷阱数量 (默认: 3)')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='训练轮数 (默认: 300)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='学习率 (默认: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='折扣因子 (默认: 0.95)')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='初始探索率 (默认: 1.0)')
    parser.add_argument('--eps-decay', type=float, default=0.995,
                        help='探索率衰减系数 (默认: 0.995)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--no-viz', action='store_true',
                        help='只训练，不显示图表')
    parser.add_argument('--save', type=str, default=None,
                        help='保存结果图的前缀路径')
    args = parser.parse_args()

    print("=" * 60)
    print("  Q-Learning 网格世界 — 教学演示")
    print("=" * 60)
    print(f"  网格大小: {args.size}×{args.size}")
    print(f"  陷阱数量: {args.traps}")
    print(f"  训练轮数: {args.episodes}")
    print(f"  学习率 α: {args.alpha}")
    print(f"  折扣因子 γ: {args.gamma}")
    print(f"  初始探索率 ε: {args.epsilon}")
    print(f"  衰减系数: {args.eps_decay}")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)

    # ---- 创建环境和 Agent ----
    env = GridWorld(size=args.size, n_traps=args.traps, seed=args.seed)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        eps_decay=args.eps_decay,
    )

    # ---- 打印初始地图 ----
    print("\n初始网格世界 (A=Agent, G=终点, X=陷阱, S=起点):\n")
    env.render_text()

    # ---- 训练 ----
    print(f"\n开始训练 ({args.episodes} 轮)...\n")
    t0 = time.time()
    history = train(env, agent, episodes=args.episodes,
                    render_every=max(1, args.episodes // 10))
    elapsed = time.time() - t0
    print(f"\n训练完成！耗时: {elapsed:.2f}s")
    print(f"最终探索率 ε: {agent.epsilon:.4f}")
    print(f"Q 表非零值占比: {(agent.q_table != 0).mean() * 100:.1f}%")

    if args.no_viz:
        return

    # ---- 可视化 ----
    print("\n生成可视化图表...")

    # 图 1: 训练曲线综合面板
    fig1 = plot_training_curves(agent, history)
    if args.save:
        fig1.savefig(f'{args.save}_training_curves.png', dpi=150, bbox_inches='tight')

    # 图 2: 网格世界 + 策略 + Q 值
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    visualize_grid_world(env, agent, ax=ax2)
    if args.save:
        fig2.savefig(f'{args.save}_grid_policy.png', dpi=150, bbox_inches='tight')

    # 图 3: 各动作 Q 值热力图
    fig3 = plot_q_table_heatmap(agent, env)
    if args.save:
        fig3.savefig(f'{args.save}_q_table.png', dpi=150, bbox_inches='tight')

    # 图 4: 最优路径
    fig4 = plot_optimal_path(env, agent)
    if args.save:
        fig4.savefig(f'{args.save}_optimal_path.png', dpi=150, bbox_inches='tight')

    plt.show()

    # ---- 最终总结 ----
    print("\n" + "=" * 60)
    print("  Q-Learning 核心要点回顾")
    print("=" * 60)
    print(f"""
  1. Q(s,a) ← Q(s,a) + α × [ r + γ × max_a' Q(s',a') - Q(s,a) ]
                                   └─────────────┘   └───────┘
                                     TD target        current Q

  2. Off-policy: 用 max Q(s',a') 更新，与当前策略无关

  3. ε-greedy 探索: 初期 ε≈1（多探索），末期 ε→0（多利用）

  4. Q 表最终收敛到最优 Q*，贪心策略即为最优策略

  超参数调优建议:
    · 收敛太慢 → 增大 α 或 减小 γ
    · 不稳定   → 减小 α
    · 探索不足 → 减小 eps_decay（衰减更慢）
    · 探索过度 → 增大 eps_decay（衰减更快）
""")
if __name__ == '__main__':
    main()