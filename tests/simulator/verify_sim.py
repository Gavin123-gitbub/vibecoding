import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from tests.simulator.scenarios import create_beam_model, generate_ema_impact
from tests.simulator.exporter import save_to_csv, save_to_uff


def main():
    # 1) 构建悬臂梁模型
    nodes, modal_params = create_beam_model()

    # 2) 运行 EMA 仿真
    time, response, force = generate_ema_impact(fs=2000, duration=2.0)

    data = {
        "time": time,
        "response": response,
        "force": force,
        "geometry": nodes,
    }

    # 3) 导出 CSV / UFF
    save_to_csv("tests/simulator/output", data)
    save_to_uff("tests/simulator/output/test_ema.uff", data)

    # 4) 绘图验证
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(time, response[0], label="Node_1")
    axes[0].set_title("Time Response (Node_1)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration")
    axes[0].legend()

    # PSD
    f, Pxx = signal.welch(response[0], fs=2000, nperseg=2048)
    axes[1].semilogy(f, Pxx, label="PSD")
    axes[1].set_xlim(0, 500)
    axes[1].set_title("PSD (Node_1)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
