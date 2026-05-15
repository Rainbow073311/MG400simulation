"""
生成 MG400 工作空间包络 STL 网格。
将 R-Z 边界轮廓绕 Z 轴旋转, 生成内/外壳体网格。
输出: workspace_cage.stl (半透明, 用于 MuJoCo 可视化)
"""
import os
import struct
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import mujoco

# 关节索引
J1, J2, J3 = 0, 1, 2
J2_RANGE = (-0.4363, 1.4835)


def _effective_j3_range_from_j2(j2):
    """J3 有效范围, 仅依赖 J2 (不依赖 data). 仅上限耦合, 下限始终为机械下限."""
    lo, hi = -0.4363, 1.8326  # URDF 默认
    if j2 < 0.0:
        j2_norm = np.clip((j2 + 0.4363) / 0.4363, 0.0, 1.0)
        hi = 1.0 + 0.833 * j2_norm
    return lo, hi


def compute_boundary(model, data, n_j2=40, n_j3=50):
    """密集采样 J2×J3 空间, 返回分层 R-Z 边界."""
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    rz_points = []
    for j2 in np.linspace(J2_RANGE[0], J2_RANGE[1], n_j2):
        lo3, hi3 = _effective_j3_range_from_j2(j2)
        for j3 in np.linspace(lo3, hi3, n_j3):
            data.qpos[:] = 0.0
            data.qpos[1] = j2
            data.qpos[2] = j3
            data.qpos[3] = -j2
            data.qpos[4] = -j3
            data.qpos[6] = j2
            data.qpos[7] = -j2
            data.qpos[8] = j3
            mujoco.mj_forward(model, data)
            pos = data.site_xpos[tcp_id]
            rz_points.append((pos[2], np.hypot(pos[0], pos[1])))

    rz = np.array(rz_points)
    z_all, r_all = rz[:, 0], rz[:, 1]

    n_layers = 50
    z_bins = np.linspace(z_all.min(), z_all.max(), n_layers + 1)
    z_centers, r_min_arr, r_max_arr = [], [], []
    for i in range(n_layers):
        mask = (z_all >= z_bins[i]) & (z_all < z_bins[i + 1])
        if mask.sum() > 10:
            z_centers.append((z_bins[i] + z_bins[i + 1]) / 2)
            r_min_arr.append(r_all[mask].min() * 0.97)   # 略微内缩
            r_max_arr.append(r_all[mask].max() * 1.01)   # 略微外扩

    return np.array(z_centers), np.array(r_min_arr), np.array(r_max_arr)


def revolve_to_mesh(z_arr, r_arr, n_theta=60, flip=False):
    """将 R-Z 轮廓绕 Z 轴旋转, 生成三角网格 (顶点 + 面). flip=True 时翻转面方向."""
    verts = []
    faces = []

    n_z = len(z_arr)
    for zi in range(n_z):
        for ti in range(n_theta):
            theta = 2 * np.pi * ti / n_theta
            x = r_arr[zi] * np.cos(theta)
            y = r_arr[zi] * np.sin(theta)
            z = z_arr[zi]
            verts.append([x, y, z])

    verts = np.array(verts)

    for zi in range(n_z - 1):
        for ti in range(n_theta):
            a = zi * n_theta + ti
            b = zi * n_theta + (ti + 1) % n_theta
            c = (zi + 1) * n_theta + ti
            d = (zi + 1) * n_theta + (ti + 1) % n_theta
            if flip:
                faces.append([a, c, b])
                faces.append([b, c, d])
            else:
                faces.append([a, b, c])
                faces.append([b, d, c])

    return verts, np.array(faces)


def annulus_mesh(z_val, r_inner, r_outer, n_theta=60, flip=False):
    """生成 Z=z_val 处的环形盖 (内外半径之间的网格)."""
    verts = []
    faces = []
    for ti in range(n_theta):
        theta = 2 * np.pi * ti / n_theta
        verts.append([r_outer * np.cos(theta), r_outer * np.sin(theta), z_val])  # 外层点
        verts.append([r_inner * np.cos(theta), r_inner * np.sin(theta), z_val])  # 内层点

    verts = np.array(verts)
    for ti in range(n_theta):
        o0 = ti * 2          # 外层当前
        i0 = ti * 2 + 1      # 内层当前
        o1 = ((ti + 1) % n_theta) * 2      # 外层下一个
        i1 = ((ti + 1) % n_theta) * 2 + 1  # 内层下一个
        if flip:
            faces.append([o0, o1, i0])
            faces.append([i0, o1, i1])
        else:
            faces.append([o0, i0, o1])
            faces.append([i0, i1, o1])

    return verts, np.array(faces)


def write_obj(filepath, verts, faces):
    """写入 OBJ 文件 (MuJoCo 原生支持)."""
    faces = np.asarray(faces)
    verts = np.asarray(verts)

    with open(filepath, "w", encoding="ascii") as f:
        f.write("# MG400 workspace cage\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            # OBJ 面索引从 1 开始
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

    size_kb = os.path.getsize(filepath) / 1024
    print(f"  OBJ 已保存: {filepath} ({len(faces)} 面, {len(verts)} 顶点, {size_kb:.0f} KB)")


def main():
    print("=" * 55)
    print("  MG400 工作空间 STL 生成器")
    print("=" * 55)

    print("  加载模型...")
    with open("MG400_urdf.xml", "r", encoding="utf-8") as f:
        xml_str = f.read()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)

    print("  密集采样 FK (40×50=2000 点)...")
    z_w, r_min, r_max = compute_boundary(model, data)
    print(f"  Z 范围: {z_w[0]*1000:.0f} ~ {z_w[-1]*1000:.0f} mm, {len(z_w)} 层")
    print(f"  R 范围: {r_min.min()*1000:.0f} ~ {r_max.max()*1000:.0f} mm")

    n_theta = 50
    r_inner = np.maximum(r_min, 0.010)

    # 仅内壳 (内边界, 表示不可达区域)
    print("  生成内壳 (不可达区域边界)...")
    v_inner, f_inner = revolve_to_mesh(z_w, r_inner, n_theta=n_theta, flip=False)

    out_path = os.path.join(SCRIPT_DIR, "urdf_meshes", "workspace_cage.obj")
    write_obj(out_path, v_inner, f_inner)

    # 保存边界数据供参考
    np.savez_compressed(
        os.path.join(SCRIPT_DIR, "workspace_cage_boundary.npz"),
        z_w=z_w, r_min=r_min, r_max=r_max,
    )
    print(f"  边界数据已保存: workspace_cage_boundary.npz")
    print("\n  完成!")


if __name__ == "__main__":
    main()
