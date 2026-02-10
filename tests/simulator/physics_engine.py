import numpy as np
from scipy import signal


class ModalSimulator:
    """
    高保真模态振动数据生成器

    geometry: [N_nodes, 3]
    modal_params: list of dicts, each:
      - freq: Hz
      - damping: xi
      - shape: vector [N_nodes]
    fs: sampling rate
    noise_level: SNR in dB (None to disable)
    """

    def __init__(self, geometry, modal_params, fs, noise_level=30.0):
        self.geometry = np.asarray(geometry, dtype=float)
        self.modal_params = modal_params
        self.fs = float(fs)
        self.noise_level = noise_level

        self._A = None
        self._B = None
        self._C = None
        self._D = None

    def _build_sys(self, input_nodes=None):
        """
        构建连续状态空间模型 (MIMO)
        状态为每阶模态的 [q, q_dot]
        输出为节点加速度 x_ddot
        """
        n_nodes = self.geometry.shape[0]
        n_modes = len(self.modal_params)

        # 组装模态矩阵 Phi [N_nodes, N_modes]
        Phi = np.zeros((n_nodes, n_modes), dtype=float)
        freqs = np.zeros(n_modes, dtype=float)
        damps = np.zeros(n_modes, dtype=float)
        for i, mp in enumerate(self.modal_params):
            shape = np.asarray(mp["shape"], dtype=float).flatten()
            if shape.shape[0] != n_nodes:
                raise ValueError("mode shape length mismatch")
            Phi[:, i] = shape
            freqs[i] = float(mp["freq"])
            damps[i] = float(mp["damping"])

        # 输入映射
        if input_nodes is None:
            input_nodes = list(range(n_nodes))
        input_nodes = list(input_nodes)
        n_inputs = len(input_nodes)

        # modal force mapping: f_modal = Phi^T * f_nodes
        # 仅取参与的输入节点
        S = np.zeros((n_nodes, n_inputs), dtype=float)
        for j, node_idx in enumerate(input_nodes):
            S[node_idx, j] = 1.0
        Fm = Phi.T @ S  # [N_modes, N_inputs]

        # 对每一阶模态构建状态方程
        # q_dot = v
        # v_dot = -w^2 q - 2 xi w v + f_modal
        w = 2.0 * np.pi * freqs

        A = np.zeros((2 * n_modes, 2 * n_modes), dtype=float)
        B = np.zeros((2 * n_modes, n_inputs), dtype=float)

        for i in range(n_modes):
            A[2 * i, 2 * i + 1] = 1.0
            A[2 * i + 1, 2 * i] = -w[i] ** 2
            A[2 * i + 1, 2 * i + 1] = -2.0 * damps[i] * w[i]
            # B for v_dot
            B[2 * i + 1, :] = Fm[i, :]

        # 输出为节点加速度: x_ddot = Phi * ( -w^2 q - 2 xi w v ) + Phi * f_modal
        C = np.zeros((n_nodes, 2 * n_modes), dtype=float)
        D = Phi @ Fm  # [N_nodes, N_inputs]

        for i in range(n_modes):
            C[:, 2 * i] = Phi[:, i] * (-w[i] ** 2)
            C[:, 2 * i + 1] = Phi[:, i] * (-2.0 * damps[i] * w[i])

        self._A, self._B, self._C, self._D = A, B, C, D
        return signal.StateSpace(A, B, C, D)

    def simulate(self, excitation_force, t, input_nodes=None):
        """
        excitation_force: [N_inputs, N_samples]
        t: time array
        返回: time, response_data [N_nodes, N_samples], force_data
        """
        if self._A is None:
            sys = self._build_sys(input_nodes=input_nodes)
        else:
            sys = signal.StateSpace(self._A, self._B, self._C, self._D)

        force = np.asarray(excitation_force, dtype=float)
        if force.ndim != 2:
            raise ValueError("excitation_force must be [N_inputs, N_samples]")
        # lsim expects [N_samples, N_inputs]
        U = force.T

        tout, y, _ = signal.lsim(sys, U=U, T=t)
        response = y.T  # [N_nodes, N_samples]

        # 加噪声: 依据 SNR (dB)
        if self.noise_level is not None:
            signal_power = np.mean(response ** 2)
            snr_linear = 10 ** (self.noise_level / 10.0)
            noise_power = signal_power / (snr_linear + 1e-12)
            noise = np.random.normal(0.0, np.sqrt(noise_power), size=response.shape)
            response = response + noise

        return tout, response, force
