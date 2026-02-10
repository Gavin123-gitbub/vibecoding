import numpy as np

from tests.simulator.physics_engine import ModalSimulator


def create_beam_model(length=1.0, num_nodes=10):
    """
    悬臂梁模型:
    - 节点沿 X 轴分布
    - 频率: [20, 120, 340] Hz
    - 阻尼: 0.01
    - 振型: 悬臂梁理论模态函数
    """
    x = np.linspace(0.0, length, num_nodes)
    nodes = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)

    freqs = [20.0, 120.0, 340.0]
    damping = 0.01

    # 悬臂梁特征根 beta_i (单位: 1/m)
    beta_vals = [1.875, 4.694, 7.855]

    modal_params = []
    for i, beta in enumerate(beta_vals):
        # sigma_i = (sinh(beta) - sin(beta)) / (cosh(beta) + cos(beta))
        sigma = (np.sinh(beta) - np.sin(beta)) / (np.cosh(beta) + np.cos(beta))
        # 归一化形状
        phi = (
            np.cosh(beta * x / length)
            - np.cos(beta * x / length)
            - sigma * (np.sinh(beta * x / length) - np.sin(beta * x / length))
        )
        phi = phi / np.max(np.abs(phi))
        modal_params.append(
            {"freq": freqs[i], "damping": damping, "shape": phi}
        )

    return nodes, modal_params


def generate_ema_impact(fs=2000, duration=2.0):
    """
    EMA 冲击测试数据:
    - 节点 9 施加脉冲力
    - 脉冲宽度 0.005s
    """
    nodes, modal_params = create_beam_model()
    t = np.arange(0, duration, 1.0 / fs)

    # excitation: node 9 (index 9)
    force = np.zeros((1, len(t)))
    pulse_len = int(0.005 * fs)
    force[0, :pulse_len] = 1.0

    sim = ModalSimulator(nodes, modal_params, fs, noise_level=30.0)
    time, response, force_data = sim.simulate(force, t, input_nodes=[9])
    return time, response, force_data


def generate_oma_operational(fs=2000, duration=5.0):
    """
    OMA 工况:
    - 节点 3, 6, 9 随机白噪声激励
    - 仅返回响应
    """
    nodes, modal_params = create_beam_model()
    t = np.arange(0, duration, 1.0 / fs)

    n_inputs = 3
    force = np.random.normal(0.0, 1.0, size=(n_inputs, len(t)))

    sim = ModalSimulator(nodes, modal_params, fs, noise_level=25.0)
    time, response, _ = sim.simulate(force, t, input_nodes=[3, 6, 9])
    return time, response


def generate_oma_transient(fs=2000, duration=5.0):
    """
    非稳态 (RDT/ERA):
    - Burst Random / Sparse Impact
    - 仅返回响应
    """
    nodes, modal_params = create_beam_model()
    t = np.arange(0, duration, 1.0 / fs)

    n_inputs = 1
    force = np.zeros((n_inputs, len(t)))

    # 随机突发噪声
    burst_count = 8
    burst_len = int(0.05 * fs)
    for _ in range(burst_count):
        start = np.random.randint(0, len(t) - burst_len)
        force[0, start : start + burst_len] += np.random.normal(0.0, 1.5, burst_len)

    # 稀疏冲击
    for _ in range(5):
        idx = np.random.randint(0, len(t))
        force[0, idx] += 3.0

    sim = ModalSimulator(nodes, modal_params, fs, noise_level=25.0)
    time, response, _ = sim.simulate(force, t, input_nodes=[9])
    return time, response
