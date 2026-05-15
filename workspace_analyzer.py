"""
MG400 可达工作空间分析器
通过采样关节空间 (J1/J2/J3), 正向运动学计算 TCP 位置,
生成可达范围的可视化图表。

用法:
    python workspace_analyzer.py
    python workspace_analyzer.py --samples 50000
"""
import argparse
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import mujoco

# 关节索引
J1, J2, J3 = 0, 1, 2

# 关节限位 (rad) — 来自 URDF
J1_RANGE = (-2.7925, 2.7925)   # ±160°
J2_RANGE = (-0.4363, 1.4835)   # -25° ~ 85°
J3_RANGE = (-0.4363, 1.8326)   # -25° ~ 105°

# J2-J3 耦合约束
def effective_j3_range(j2):
    lo, hi = J3_RANGE
    if j2 < 0.0:
        lo = max(J3_RANGE[0], j2)        # J3 下限跟随 J2
        j2_norm = np.clip((j2 + 0.4363) / 0.4363, 0.0, 1.0)
        hi = 1.0 + 0.833 * j2_norm       # J2=-25°→1.0,  J2=0°→1.833
    return lo, hi


def load_model():
    xml_path = os.path.join(SCRIPT_DIR, "MG400_urdf.xml")
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_str = f.read()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    return model, data


def sample_workspace(model, data, n_samples=30000):
    """随机采样关节空间, 返回所有可达 TCP 位置 (世界坐标, 米)。"""
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    points = np.zeros((n_samples, 3))
    valid = 0

    for i in range(n_samples):
        # 均匀采样 J1, J2
        j1 = np.random.uniform(*J1_RANGE)
        j2 = np.random.uniform(*J2_RANGE)

        # J3 受 J2 耦合约束
        j3_lo, j3_hi = effective_j3_range(j2)
        j3 = np.random.uniform(j3_lo, j3_hi)

        # 设置 qpos: J1, J2, J3, J31, J41, J4, J22, J32, J42
        data.qpos[:] = 0.0
        data.qpos[0] = j1
        data.qpos[1] = j2
        data.qpos[2] = j3
        # mimic joints
        data.qpos[3] = -j2   # J31 = -J2
        data.qpos[4] = -j3   # J41 = -J3
        data.qpos[6] = j2    # J22 = J2
        data.qpos[7] = -j2   # J32 = -J2
        data.qpos[8] = j3    # J42 = J3

        mujoco.mj_forward(model, data)
        points[valid] = data.site_xpos[tcp_id]
        valid += 1

    return points[:valid]


def compute_workspace_stats(points_m):
    """计算工作空间统计信息 (mm)。"""
    p = points_m * 1000
    r = np.hypot(p[:, 0], p[:, 1])
    return {
        "X": (p[:, 0].min(), p[:, 0].max()),
        "Y": (p[:, 1].min(), p[:, 1].max()),
        "Z": (p[:, 2].min(), p[:, 2].max()),
        "R": (r.min(), r.max()),
        "count": len(p),
    }


def generate_plots(points_m, out_dir):
    """生成工作空间可视化图表。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = points_m * 1000  # 转毫米
    r = np.hypot(p[:, 0], p[:, 1])

    os.makedirs(out_dir, exist_ok=True)

    # ── 图1: R-Z 剖面 (径向距离 vs 高度, 最关键) ──
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(r, p[:, 2], s=0.3, c="steelblue", alpha=0.3, rasterized=True)
    ax.set_xlabel("径向距离 r = sqrt(x²+y²)  [mm]")
    ax.set_ylabel("Z 高度 [mm]")
    ax.set_title("MG400 可达工作空间 — R-Z 剖面")
    ax.grid(True, alpha=0.3)

    # 标注不可达区域 (Z=-26mm相对, 但这里显示世界坐标, 所以用初始Z-26≈164mm)
    # 不做硬编码, 只标注边界
    ax.axhline(y=p[:, 2].min(), color="red", linestyle="--", alpha=0.5, label=f"Zmin={p[:,2].min():.0f}mm")
    ax.axhline(y=p[:, 2].max(), color="red", linestyle="--", alpha=0.5, label=f"Zmax={p[:,2].max():.0f}mm")
    ax.axvline(x=r.max(), color="orange", linestyle="--", alpha=0.5, label=f"Rmax={r.max():.0f}mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "workspace_RZ.png"), dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_dir}/workspace_RZ.png")

    # ── 图2: X-Y 俯视图 ──
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(p[:, 0], p[:, 1], s=0.3, c="steelblue", alpha=0.3, rasterized=True)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_title("MG400 可达工作空间 — X-Y 俯视图")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    # 画最大/最小半径圆
    for rad, label, color in [(r.min(), f"Rmin={r.min():.0f}", "red"),
                                (r.max(), f"Rmax={r.max():.0f}", "orange")]:
        circle = plt.Circle((0, 0), rad, fill=False, color=color, linestyle="--", alpha=0.7)
        ax.add_patch(circle)
        ax.annotate(label, (0, rad), textcoords="offset points", xytext=(5, 5), fontsize=8, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "workspace_XY.png"), dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_dir}/workspace_XY.png")

    # ── 图3: X-Z 剖面 (Y≈0 切片) ──
    mask_y0 = np.abs(p[:, 1]) < 30  # Y 在 ±30mm 内
    if mask_y0.sum() > 100:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(p[mask_y0, 0], p[mask_y0, 2], s=0.5, c="steelblue", alpha=0.4, rasterized=True)
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Z [mm]")
        ax.set_title("MG400 可达工作空间 — X-Z 剖面 (|Y|<30mm)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "workspace_XZ.png"), dpi=150)
        plt.close(fig)
        print(f"  已保存: {out_dir}/workspace_XZ.png")

    # ── 图4: Z 分层 R 范围 ──
    fig, ax = plt.subplots(figsize=(10, 7))
    z_bins = np.linspace(p[:, 2].min(), p[:, 2].max(), 50)
    r_min_per_z = []
    r_max_per_z = []
    z_centers = []
    for i in range(len(z_bins) - 1):
        mask = (p[:, 2] >= z_bins[i]) & (p[:, 2] < z_bins[i + 1])
        if mask.sum() > 10:
            r_min_per_z.append(r[mask].min())
            r_max_per_z.append(r[mask].max())
            z_centers.append((z_bins[i] + z_bins[i + 1]) / 2)

    ax.fill_betweenx(z_centers, r_min_per_z, r_max_per_z, alpha=0.3, color="steelblue")
    ax.plot(r_min_per_z, z_centers, "b-", linewidth=1, alpha=0.7)
    ax.plot(r_max_per_z, z_centers, "r-", linewidth=1, alpha=0.7, label="R 最大边界")
    ax.set_xlabel("径向距离 r [mm]")
    ax.set_ylabel("Z 高度 [mm]")
    ax.set_title("MG400 可达工作空间 — 各 Z 层的 R 范围")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "workspace_R_by_Z.png"), dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_dir}/workspace_R_by_Z.png")

    # ── 图5: 3D 散点 (轻量级, 固定视角) ──
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    # 降采样以提高渲染速度
    n_show = min(len(p), 15000)
    idx = np.random.choice(len(p), n_show, replace=False)
    ax.scatter(p[idx, 0], p[idx, 1], p[idx, 2], s=0.2, c=p[idx, 2], cmap="viridis", alpha=0.5)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("MG400 可达工作空间 — 3D 点云")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "workspace_3D.png"), dpi=150)
    plt.close(fig)
    print(f"  已保存: {out_dir}/workspace_3D.png")


def main():
    parser = argparse.ArgumentParser(description="MG400 工作空间分析器")
    parser.add_argument("--samples", type=int, default=30000, help="采样数 (默认 30000)")
    parser.add_argument("--out", default="workspace_output", help="输出目录")
    args = parser.parse_args()

    print("=" * 55)
    print("  MG400 可达工作空间分析器")
    print("=" * 55)
    print(f"  采样数: {args.samples}")
    print(f"  加载模型...")

    model, data = load_model()

    print(f"  关节限位:")
    print(f"    J1: {np.degrees(J1_RANGE[0]):.0f}° ~ {np.degrees(J1_RANGE[1]):.0f}°")
    print(f"    J2: {np.degrees(J2_RANGE[0]):.0f}° ~ {np.degrees(J2_RANGE[1]):.0f}°")
    print(f"    J3: {np.degrees(J3_RANGE[0]):.0f}° ~ {np.degrees(J3_RANGE[1]):.0f}°  (有 J2-J3 耦合)")

    print(f"  采样中...")
    points = sample_workspace(model, data, args.samples)
    print(f"  有效采样: {len(points)} 点")

    stats = compute_workspace_stats(points)
    print(f"\n  工作空间统计 (世界坐标, mm):")
    print(f"    X:  {stats['X'][0]:.0f} ~ {stats['X'][1]:.0f}")
    print(f"    Y:  {stats['Y'][0]:.0f} ~ {stats['Y'][1]:.0f}")
    print(f"    Z:  {stats['Z'][0]:.0f} ~ {stats['Z'][1]:.0f}")
    print(f"    R:  {stats['R'][0]:.0f} ~ {stats['R'][1]:.0f}")

    print(f"\n  生成可视化图表...")
    generate_plots(points, args.out)

    # 保存原始数据
    np.savez_compressed(
        os.path.join(args.out, "workspace_points.npz"),
        points_m=points * 1000,
        stats=stats,
    )
    print(f"  已保存: {args.out}/workspace_points.npz")
    print(f"\n  完成! 共 {len(points)} 个可达点.")


if __name__ == "__main__":
    main()
