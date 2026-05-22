"""
MG400 机械臂 MuJoCo 仿真 — 滑块控制 + 鼠标拖拽 (逆运动学)
用法:
    python simulate_slider.py
    python simulate_slider.py --model MG400_urdf.xml

操作:
    滑块模式 (默认): 左侧面板拖动 J1/J2/J3/J4 滑块控制关节
    自由拖拽模式:   按 Tab 切换, Ctrl+右键拖拽末端 (IK 实时解算)
"""
import argparse
import ctypes
import json
import os
import socket
import sys
import threading

# Windows 控制台 UTF-8 支持 (解决 GBK 编码报错)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import glfw
import numpy as np
import mujoco
import mujoco.viewer

import mg400_server

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 4 个被驱动关节在 qpos 中的索引
ACTUATED_QPOS = [0, 1, 2, 5]  # J1, J2, J3, J4

# 位置 IK 仅使用 J1/J2/J3, J4 绕自身 Z 轴旋转不影响 TCP 位置
IK_JOINTS = [0, 1, 2]

# 从动关节 (mimic) 映射: (目标qpos索引, 源qpos索引, 系数)
MIMIC_MAP = [
    (3, 1, -1),   # J31 = -J2
    (4, 2, -1),   # J41 = -J3
    (6, 1,  1),   # J22 = J2
    (7, 1, -1),   # J32 = -J2
    (8, 2,  1),   # J42 = J3
]


def load_model(xml_name="MG400_urdf.xml"):
    xml_path = os.path.join(SCRIPT_DIR, xml_name)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"模型文件不存在: {xml_path}")
    os.chdir(SCRIPT_DIR)
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_str = f.read()
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    return model, data


def _sync_mimic_joints(model, data):
    """将从动关节的位置同步到主关节一致, 并确保不超限位。"""
    for tgt, src, coeff in MIMIC_MAP:
        val = coeff * data.qpos[src]
        lo, hi = model.jnt_range[tgt]
        data.qpos[tgt] = np.clip(val, lo, hi)


def _effective_joint_range(model, data, qi):
    """关节有效限位 — J3 上限与 J2 耦合 (并联连杆机构在 J2 低位时干涉)"""
    lo, hi = model.jnt_range[qi]
    if qi == 2:  # J3
        j2 = data.qpos[1]
        if j2 < 0.0:
            # J3 上限: J2=-25°→1.0rad, J2=0°→1.833rad
            j2_norm = np.clip((j2 + 0.4363) / 0.4363, 0.0, 1.0)
            hi = 1.0 + 0.833 * j2_norm
    return lo, hi


def _compute_waypoints(start_xyz, target_xyz, seg_m=0.05):
    """线性插值, 每段约 seg_m 米, 返回路径点列表 (不含起点)。"""
    s = np.asarray(start_xyz, dtype=float)
    t = np.asarray(target_xyz, dtype=float)
    d = t - s
    dist = np.linalg.norm(d)
    n = max(1, int(round(dist / seg_m)))
    return [(s + d * (i / n)).tolist() for i in range(1, n + 1)]


def _toggle_workspace_vis(model):
    """开关工作空间 cage 的可见性 (通过材质 alpha 通道)。"""
    mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "mat_workspace")
    if mat_id < 0:
        return False
    rgba = model.mat_rgba[mat_id]
    if rgba[3] > 0.01:
        model.mat_rgba[mat_id, 3] = 0.0
        return False
    else:
        model.mat_rgba[mat_id, 3] = 0.40
        return True


def _in_unreachable_zone(point, z_initial):
    """检查 3D 点是否在不可达区域: Z=-26mm(相对), 距 Z 轴 r<190mm."""
    if z_initial is None:
        return False
    unreach_z = z_initial - 0.026  # 世界 Z 坐标
    r = np.hypot(point[0], point[1])
    return abs(point[2] - unreach_z) < 0.005 and r < 0.190


def _segment_crosses_unreachable(p1, p2, z_initial):
    """检查线段 p1→p2 是否穿过不可达圆盘 (Z=-26mm相对, r<190mm)."""
    if z_initial is None:
        return False
    unreach_z = z_initial - 0.026
    z1, z2 = p1[2], p2[2]
    if (z1 - unreach_z) * (z2 - unreach_z) >= 0:
        return False  # 两端在同侧, 不穿过平面
    # 线段与 Z=-26mm 平面的交点
    t = (unreach_z - z1) / (z2 - z1)
    x_int = p1[0] + t * (p2[0] - p1[0])
    y_int = p1[1] + t * (p2[1] - p1[1])
    return np.hypot(x_int, y_int) < 0.190


def _plan_safe_motion(cur, target_xyz, descent_seg_m=0.001, verbose_prefix=""):
    """四阶段安全运动规划: 抬升 → J1旋转 → XY平移 → 寸动下降。
    短距离移动 (<40mm) 直接走直线, 跳过安全抬升。
    返回路径点列表 (不含起点)。"""
    wp_list = []
    SAFE_MARGIN = 0.080
    x, y, world_z = target_xyz
    tgt = np.array(target_xyz, dtype=float)

    # ── 短距直连: 总距离 <40mm 直接走直线 ──
    total_dist = np.linalg.norm(tgt - np.asarray(cur, dtype=float))
    if total_dist < 0.040:
        segs = _compute_waypoints(cur, target_xyz, seg_m=0.005)
        wp_list.extend(segs)
        if verbose_prefix:
            print(f"  {verbose_prefix}短距直连: {total_dist*1000:.0f}mm, {len(segs)}段")
        return wp_list

    safe_z = max(cur[2], world_z) + SAFE_MARGIN
    tgt_r = np.hypot(x, y)
    tgt_th = np.arctan2(y, x)

    # 阶段1: 抬升 Z 到安全高度
    if abs(cur[2] - safe_z) > 0.002:
        segs = _compute_waypoints(cur, [cur[0], cur[1], safe_z])
        wp_list.extend(segs)
        if verbose_prefix:
            print(f"  {verbose_prefix}阶段1 抬升Z: {cur[2]*1000:.0f}→{safe_z*1000:.0f}mm | {len(segs)}段")

    # 阶段2a: 旋转 J1 到目标方位
    cur_r = np.hypot(cur[0], cur[1])
    cur_th = np.arctan2(cur[1], cur[0])
    if abs(np.arctan2(np.sin(tgt_th - cur_th), np.cos(tgt_th - cur_th))) > 0.02:
        mid_x = cur_r * np.cos(tgt_th)
        mid_y = cur_r * np.sin(tgt_th)
        segs = _compute_waypoints([cur[0], cur[1], safe_z], [mid_x, mid_y, safe_z])
        wp_list.extend(segs)
        if verbose_prefix:
            print(f"  {verbose_prefix}阶段2a J1旋转: θ={np.degrees(cur_th):.0f}°→{np.degrees(tgt_th):.0f}° | {len(segs)}段")
    else:
        mid_x, mid_y = cur[0], cur[1]

    # 阶段2b: XY 平移到目标正上方
    if abs(mid_x - x) > 0.002 or abs(mid_y - y) > 0.002:
        segs = _compute_waypoints([mid_x, mid_y, safe_z], [x, y, safe_z])
        wp_list.extend(segs)
        if verbose_prefix:
            print(f"  {verbose_prefix}阶段2b XY平移: r={cur_r*1000:.0f}→{tgt_r*1000:.0f}mm | {len(segs)}段")

    # 阶段3: 下降 — 上半段5mm步长, 末段50mm切回寸动1mm
    descent_z_dist = abs(safe_z - world_z)
    if descent_z_dist > 0.005:
        if descent_z_dist > 0.050:
            # 上半段: 5mm步长降到目标上方50mm
            mid_z = world_z + 0.050 * (1 if world_z > safe_z else -1)
            segs = _compute_waypoints([x, y, safe_z], [x, y, mid_z], seg_m=0.005)
            wp_list.extend(segs)
            # 下半段: 1mm寸动
            segs = _compute_waypoints([x, y, mid_z], [x, y, world_z], seg_m=0.001)
            wp_list.extend(segs)
            if verbose_prefix:
                print(f"  {verbose_prefix}阶段3 下降: {safe_z*1000:.0f}→{mid_z*1000:.0f}(5mm/段)→{world_z*1000:.0f}(1mm寸动) | {len(segs)}段")
        else:
            segs = _compute_waypoints([x, y, safe_z], [x, y, world_z], seg_m=0.001)
            wp_list.extend(segs)
            if verbose_prefix:
                print(f"  {verbose_prefix}阶段3 下降Z: {safe_z*1000:.0f}→{world_z*1000:.0f}mm(1mm寸动) | {len(segs)}段")
    elif abs(cur[2] - world_z) > 0.001:
        # 无需抬升, 直接下降
        direct_dz = abs(cur[2] - world_z)
        seg_m_fast = 0.005 if direct_dz > 0.050 else 0.001
        segs = _compute_waypoints([cur[0], cur[1], cur[2]], [x, y, world_z], seg_m=seg_m_fast)
        wp_list.extend(segs)
        if verbose_prefix:
            print(f"  {verbose_prefix}阶段3 直接下降: {cur[2]*1000:.0f}→{world_z*1000:.0f}mm | {len(segs)}段")

    return wp_list


# ── 障碍物检测与避障 ──

def _get_robot_keypoints(data, tcp_site_id, model):
    """获取机器人关键点世界坐标: TCP, 肘部, 腕部。"""
    tcp = data.site_xpos[tcp_site_id].copy()
    # 肘部近似: fake_link 的 body id
    elbow_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fake_link")
    elbow = data.xpos[elbow_id].copy() if elbow_id >= 0 else tcp.copy()
    # 腕部近似: link4_1 body
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link4_1")
    wrist = data.xpos[wrist_id].copy() if wrist_id >= 0 else tcp.copy()
    return {"tcp": tcp, "elbow": elbow, "wrist": wrist}


def _point_to_segment_dist(p, a, b):
    """点到线段的最短距离 (世界坐标)。"""
    ab = b - a
    ap = p - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab) if np.dot(ab, ab) > 1e-12 else 0.0, 0.0, 1.0)
    return np.linalg.norm(ap - t * ab)


def _obstacle_segment(obs_pos, half_len):
    """障碍物线段端点 (世界坐标): 水平沿X轴, obs_pos为中心。"""
    a = obs_pos + np.array([-half_len, 0.0, 0.0])
    b = obs_pos + np.array([ half_len, 0.0, 0.0])
    return a, b


def _check_obstacle_collision(tcp_pos, obs_pos, obs_radius, half_len, safe_dist=0.050):
    """检查 TCP 是否进入障碍物安全范围 (50mm)。
    返回 (collision_bool, surface_distance_m)。"""
    a, b = _obstacle_segment(obs_pos, half_len)
    seg_dist = _point_to_segment_dist(tcp_pos, a, b)
    surf_dist = max(0.0, seg_dist - obs_radius)
    return surf_dist < safe_dist, surf_dist


# ── Win32 鼠标轮询（不设 GLFW 回调，不与 viewer 冲突）──
_user32 = ctypes.windll.user32


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_cursor_screen():
    pt = _POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def _is_right_pressed():
    return _user32.GetAsyncKeyState(0x02) & 0x8000 != 0


def _is_ctrl_pressed():
    return _user32.GetAsyncKeyState(0x11) & 0x8000 != 0


# ── 外部控制 TCP 服务 ──
Z_INITIAL = None          # 初始 TCP 世界 Z 坐标, 作为相对坐标系原点
_HOME_TCP = None          # 原点姿态 TCP 世界坐标 (J1=J2=J3=J4=0)
_WS_Z_WORLD = None        # 工作空间边界 Z 数组 (世界坐标, m)
_WS_R_MIN = None          # 工作空间边界 R_min 数组 (世界坐标, m)
_WS_R_MAX = None          # 工作空间边界 R_max 数组 (世界坐标, m)

# 外部控制共享状态 — 主循环与 mg400_server 通过同一对象通信
from types import SimpleNamespace as _Ns
ext = _Ns()
ext.lock = threading.Lock()
ext.target = None         # [x, y, z] 最终目标 (世界坐标)
ext.active = False        # 外部目标是否激活
ext.waypoints = None      # [[x,y,z], ...] 插值路径点
ext.wp_idx = -1           # 当前路径点索引 (-1=无)
ext.speed = 0.3           # 移动速度 [0.05, 1.0], 默认 0.3
ext.tcp = np.zeros(3)     # 当前 TCP 位置 (供查询用)
ext.error = 0.0           # 当前位置误差 (到当前路径点)
ext.final_error = 0.0     # 当前位置误差 (到最终目标, 供控制面板显示)
ext.reached = False       # 外部目标是否已完成
ext.joint_move = False    # 是否关节模式运动
ext.target_joints = None  # 目标关节角 rad [j1,j2,j3,j4]
ext.obstacle_active = False  # 避障开关
ext.obstacle_pos = np.zeros(3)  # 障碍物世界坐标 [x,y,z]
ext.obstacle_radius = 0.025    # 障碍物圆柱半径 (m)
ext.obstacle_half_length = 0.12 # 障碍物半长 (m), 沿局部X轴
ext.collision_info = None    # 最近一次碰撞信息 dict 或 None

def _tcp_server():
    """MG400 协议服务器已在 mg400_server.py 中, 此函数已移除。"""


def main():
    parser = argparse.ArgumentParser(description="MG400 滑块控制仿真")
    parser.add_argument("--model", default="MG400_urdf.xml")
    args = parser.parse_args()

    model, data = load_model(args.model)
    mujoco.mj_forward(model, data)

    print("=" * 55)
    print(f"  MG400 URDF 滑块控制 - {model.nq} DOF")
    print("=" * 55)
    print(f"  关节数: {model.njnt}")
    print(f"  执行器数 (position滑块): {model.nu}")
    print(f"  几何体数: {model.ngeom}")
    print("  网格来源: HarvestX/MG400_ROS2 URDF")
    print("-" * 55)
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        print(f"  [{i}] {name}: axis={model.jnt_axis[i]}, "
              f"range={model.jnt_range[i]}")
    print("-" * 55)

    # ── 按键回调相关（必须在 launch_passive 之前定义, 作为参数传入）──
    free_drag = [False]
    show_workspace = [True]
    selected_joint = [0]  # 0=J1, 1=J2, 2=J3, 3=J4
    orig_gainprm = model.actuator_gainprm.copy()

    def enter_drag_mode():
        for i in range(model.nu):
            model.actuator_gainprm[i, 0] = 0.0
            model.actuator_gainprm[i, 1] = 0.0
        print("\n  >>> 自由拖拽模式 (Ctrl+右键拖拽末端, IK 解算) <<<\n")

    def enter_slider_mode():
        model.actuator_gainprm[:] = orig_gainprm
        print("\n  >>> 滑块控制模式 <<<\n")

    def key_callback_ext(key):
        """按键回调：Tab 切换模式 + 数字键选关节 + O 障碍物 + W 工作空间
        MuJoCo 3.8 key_callback 签名为 Callable[[int], None], 仅在 PRESS 时触发。"""
        if key == glfw.KEY_TAB:
            free_drag[0] = not free_drag[0]
            if free_drag[0]:
                enter_drag_mode()
            else:
                enter_slider_mode()
        if key == glfw.KEY_W:
            show_workspace[0] = _toggle_workspace_vis(model)
            print(f"  工作空间可视化: {'ON' if show_workspace[0] else 'OFF'}")
        elif key == glfw.KEY_O:
            ext.obstacle_active = not ext.obstacle_active
            obs_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
            if ext.obstacle_active:
                if obs_geom_id >= 0:
                    model.geom_rgba[obs_geom_id] = [1.0, 0.2, 0.1, 0.55]
                print(f"  障碍物避障: ON (O键切换, 方向键移动)")
            else:
                if obs_geom_id >= 0:
                    model.geom_rgba[obs_geom_id] = [1.0, 0.2, 0.1, 0.0]
                print(f"  障碍物避障: OFF")
        for i, k in enumerate([glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4]):
            if key == k:
                selected_joint[0] = i
                print(f"  选中关节: J{i+1}")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True,
        key_callback=key_callback_ext,
    ) as viewer:
        if hasattr(viewer, 'cam'):
            viewer.cam.lookat[:] = [0.0, 0.0, 0.25]
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -30

        data.qpos[:] = 0.0
        data.ctrl[:] = 0.0
        mujoco.mj_forward(model, data)

        # ── J1 执行器滑块范围收紧到 ±160° ──
        # 关节限位安全余量 — 防止贴死限位导致不可逆卡死
        JNT_MARGIN = 0.005
        model.actuator_ctrlrange[0, 0] = np.clip(model.jnt_range[0, 0] + JNT_MARGIN, -2.7925, None)
        model.actuator_ctrlrange[0, 1] = np.clip(model.jnt_range[0, 1] - JNT_MARGIN, None, 2.7925)
        model.actuator_ctrlrange[1, 0] = model.jnt_range[1, 0] + JNT_MARGIN  # J2 下限 +余量
        model.actuator_ctrlrange[1, 1] = model.jnt_range[1, 1] - JNT_MARGIN  # J2 上限 -余量
        model.actuator_ctrlrange[2, 0] = model.jnt_range[2, 0] + JNT_MARGIN  # J3 下限 +余量
        model.actuator_ctrlrange[2, 1] = model.jnt_range[2, 1] - JNT_MARGIN  # J3 上限 -余量

        # ── TCP 站点 ──
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if tcp_site_id < 0:
            raise RuntimeError("TCP site 'tcp' not found in model")
        print(f"  TCP site: [{tcp_site_id}] tcp")

        target_marker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        if target_marker_body_id < 0:
            print("  警告: target_marker body 未找到")
            target_marker_mocap_id = -1
        else:
            target_marker_mocap_id = model.body_mocapid[target_marker_body_id]
            print(f"  Target marker: body_id={target_marker_body_id}, mocap_id={target_marker_mocap_id}")

        # 障碍物 mocap 初始化
        obs_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        if obs_body_id >= 0:
            obs_mocap_id = model.body_mocapid[obs_body_id]
            # 默认位置: 工作空间中部偏左, 模拟操作员手臂
            data.mocap_pos[obs_mocap_id] = [0.28, -0.08, 0.28]
            ext.obstacle_pos[:] = data.mocap_pos[obs_mocap_id]
            print(f"  障碍物: body_id={obs_body_id}, mocap_id={obs_mocap_id} (按O键激活)")
        else:
            obs_mocap_id = -1
            print(f"  警告: obstacle body 未找到")

        # 初始化外部状态 + 捕获原点姿态
        ext.tcp[:] = data.site_xpos[tcp_site_id]
        global Z_INITIAL, _HOME_TCP, _WS_Z_WORLD, _WS_R_MIN, _WS_R_MAX
        Z_INITIAL = float(data.site_xpos[tcp_site_id][2])
        _HOME_TCP = data.site_xpos[tcp_site_id].copy()  # 原点姿态 TCP 位置 (J1=J2=J3=J4=0时)
        print(f"  Z_INITIAL (世界坐标): {Z_INITIAL*1000:.1f} mm (以此为 Z=0 的相对坐标系原点)")
        print(f"  HOME TCP (世界): {_HOME_TCP}")

        # 加载工作空间边界 (供控制面板验证用)
        try:
            bnd = np.load(os.path.join(SCRIPT_DIR, "workspace_cage_boundary.npz"))
            _WS_Z_WORLD = bnd["z_w"].astype(float)  # 世界坐标, m
            _WS_R_MIN = bnd["r_min"].astype(float)   # m
            _WS_R_MAX = bnd["r_max"].astype(float)   # m
            print(f"  工作空间边界已加载: {len(_WS_Z_WORLD)} 层")
        except FileNotFoundError:
            print("  警告: 未找到 workspace_cage_boundary.npz, 控制面板验证不可用")

        # ── 构建共享状态 (供 mg400_server Dashboard/Feedback 使用) ──
        server_state = {
            "model": model, "data": data, "tcp_site_id": tcp_site_id,
            "ext": ext,
            "Z_INITIAL": Z_INITIAL, "_HOME_TCP": _HOME_TCP,
            "_plan_safe_motion": _plan_safe_motion,
            "_compute_waypoints": _compute_waypoints,
            "_in_unreachable_zone": _in_unreachable_zone,
            "_segment_crosses_unreachable": _segment_crosses_unreachable,
        }

        # 启动 MG400 协议服务器 (Dashboard 29999 + Feedback 30004)
        mg400_server.start_servers(server_state)

        grav_comp = np.zeros(model.nv)
        enter_slider_mode()

        # ── 鼠标拖拽状态（Win32 轮询）──
        drag = {"active": False, "last_sx": 0, "last_sy": 0,
                "target": np.zeros(3)}
        headpos = np.zeros(3)
        forward = np.zeros(3)
        up_vec = np.zeros(3)
        right_vec = np.zeros(3)

        # Jacobian 缓冲区 (3 × nv)
        jac_buf = np.zeros((3, model.nv))

        # J4 末端法兰盘 — 外壳已固定于 link4, 仅法兰旋转

        print("  Tab 键: 切换 滑块控制 / 自由拖拽")
        print("  W  键: 切换 工作空间包络线")
        print("  O  键: 切换 障碍物避障 (操作员手臂仿真)")
        print("  障碍物移动: ↑↓←→ 移动XY, PageUp/PageDown 移动Z")
        print("  拖拽操作: Ctrl+右键拖拽末端 (IK 逆运动学)")
        print("  方向键: ↑↓ 微调关节 (仅自由拖拽+障碍物关闭时)")
        print("-" * 55)

        # ext is module-level, attributes accessed without global
        was_soft = free_drag[0]  # 初始软模式状态
        stall_frames = 0         # 路径点停滞计数 (帧)
        best_dist = 999.0        # 当前路径点最佳误差 (m)
        current_wp_list = None   # 当前路径点列表 (用于检测变更)
        current_wp_idx = -1      # 当前路径点索引
        unreachable_cooldown = 0 # 不可到达提示冷却帧数
        stall_cycles = 0         # 连续停滞周期计数
        replan_count = 0         # 连续重算路径计数
        # 障碍物键盘边沿检测: 仅在按键从 0→1 时触发 (每次 20mm)
        _OBS_KEY_MAP = [
            (0x26, ( 0,  1,  0)),  # ↑ → 世界+Y
            (0x28, ( 0, -1,  0)),  # ↓ → 世界-Y
            (0x25, (-1,  0,  0)),  # ← → 世界-X
            (0x27, ( 1,  0,  0)),  # → → 世界+X
            (0x21, ( 0,  0,  1)),  # PageUp → +Z
            (0x22, ( 0,  0, -1)),  # PageDown → -Z
        ]
        _obs_key_prev = {vk: False for vk, _ in _OBS_KEY_MAP}
        while viewer.is_running():
            # ── 检测外部目标 ──
            ext_now = False
            with ext.lock:
                ext_now = ext.active
            soft_mode = free_drag[0] or ext_now

            # ── 软模式进入/退出时切换执行器增益 ──
            if soft_mode and not was_soft:
                for i in range(model.nu):
                    model.actuator_gainprm[i, 0] = 0.0
                    model.actuator_gainprm[i, 1] = 0.0
            elif not soft_mode and was_soft:
                # 恢复前同步 ctrl 到当前 qpos 并清零速度, 避免回弹
                for i, qidx in enumerate(ACTUATED_QPOS):
                    data.ctrl[i] = data.qpos[qidx]
                    data.qvel[qidx] = 0.0
                model.actuator_gainprm[:] = orig_gainprm

            if soft_mode:

                # ── 处理关节模式运动 (MovJ joint) → FK → TCP 路径 ──
                if ext.joint_move and ext.target_joints is not None:
                    with ext.lock:
                        if ext.joint_move and ext.target_joints is not None:
                            j_rad = ext.target_joints
                            ext.joint_move = False
                            ext.target_joints = None
                            cur = ext.tcp.tolist()
                            # FK: 临时设置关节角 → mj_forward → 读 TCP
                            save = data.qpos.copy()
                            data.qpos[0] = j_rad[0]
                            data.qpos[1] = j_rad[1]
                            data.qpos[2] = j_rad[2]
                            data.qpos[5] = j_rad[3]
                            data.qpos[3] = -j_rad[1]
                            data.qpos[4] = -j_rad[2]
                            data.qpos[6] = j_rad[1]
                            data.qpos[7] = -j_rad[1]
                            data.qpos[8] = j_rad[2]
                            mujoco.mj_forward(model, data)
                            tcp_world = data.site_xpos[tcp_site_id].copy()
                            data.qpos[:] = save
                            mujoco.mj_forward(model, data)
                            # 规划安全路径
                            wp_list = _plan_safe_motion(cur, tcp_world.tolist(),
                                                        verbose_prefix="Joint→ ")
                            ext.target = tcp_world.tolist()
                            ext.waypoints = wp_list if wp_list else None
                            ext.wp_idx = 0 if ext.waypoints else -1
                            ext.active = True
                            ext.reached = False
                            print(f"  Joint→FK TCP: [{tcp_world[0]*1000:.0f},{tcp_world[1]*1000:.0f},{tcp_world[2]*1000:.0f}]mm, {len(wp_list)}段")

                # ── 鼠标拖拽 (仅自由拖拽模式) ──
                if free_drag[0]:
                    if _is_right_pressed() and _is_ctrl_pressed():
                        sx, sy = _get_cursor_screen()
                        if not drag["active"]:
                            drag["active"] = True
                            drag["last_sx"] = sx
                            drag["last_sy"] = sy
                            _sync_mimic_joints(model, data)
                            mujoco.mj_forward(model, data)
                            drag["target"][:] = data.site_xpos[tcp_site_id]
                            print(f"  拖拽起点: {drag['target']}")
                        else:
                            dsx = sx - drag["last_sx"]
                            dsy = sy - drag["last_sy"]
                            drag["last_sx"] = sx
                            drag["last_sy"] = sy
                            if dsx != 0 or dsy != 0:
                                mujoco.mjv_cameraFrame(
                                    headpos, forward, up_vec, right_vec,
                                    data, viewer.cam)
                                vp = viewer.viewport
                                pixel_scale = viewer.cam.distance / max(vp[2], 1) * 0.8
                                drag["target"] += (dsx * right_vec - dsy * up_vec) * pixel_scale
                    else:
                        if drag["active"]:
                            drag["active"] = False
                            print("  拖拽结束")

                # ── 确定 IK 目标 (鼠标拖拽优先, 其次外部目标) ──
                ik_target = None
                speed = 1.0  # 鼠标拖拽全速
                if free_drag[0] and drag["active"]:
                    ik_target = drag["target"]
                elif ext_now:
                    with ext.lock:
                        wp_list = ext.waypoints
                        wp_idx = ext.wp_idx
                        speed = ext.speed
                    if wp_list is not None and 0 <= wp_idx < len(wp_list):
                        ik_target = np.array(wp_list[wp_idx], dtype=float)

                # ── IK 解算 ──
                if ik_target is not None:
                    _sync_mimic_joints(model, data)
                    mujoco.mj_forward(model, data)
                    tcp_pos = data.site_xpos[tcp_site_id]
                    error = ik_target - tcp_pos
                    dist = np.linalg.norm(error)

                    if dist > 1e-6:
                        mujoco.mj_jacSite(model, data, jac_buf, None, tcp_site_id)
                        # 3×3 有效雅可比 (含所有从动关节: J31/J41/J22/J32/J42)
                        J = np.zeros((3, 3))
                        J[:, 0] = jac_buf[:, 0]                                                       # J1
                        J[:, 1] = jac_buf[:, 1] - jac_buf[:, 3] + jac_buf[:, 6] - jac_buf[:, 7]       # J2 + mimics
                        J[:, 2] = jac_buf[:, 2] - jac_buf[:, 4] + jac_buf[:, 8]                        # J3 + mimics

                        # ── 障碍物触碰检测 (安全距离 50mm, 触碰则暂停) ──
                        if ext.obstacle_active and obs_mocap_id >= 0:
                            obs_pos = ext.obstacle_pos
                            collided, surf_dist = _check_obstacle_collision(
                                tcp_pos, obs_pos, ext.obstacle_radius, ext.obstacle_half_length)
                            # 障碍物可视化: 靠近时变红
                            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
                            if geom_id >= 0:
                                if surf_dist < 0.050:
                                    danger = 1.0 - surf_dist / 0.050
                                    model.geom_rgba[geom_id] = [1.0, 0.2 * (1 - danger), 0.1 * (1 - danger),
                                                                 0.55 + danger * 0.45]
                                else:
                                    model.geom_rgba[geom_id] = [1.0, 0.2, 0.1, 0.55]
                            # 触碰 → 暂停运动, 清除路径点
                            if collided and ext.active:
                                tcp_world = data.site_xpos[tcp_site_id].copy()
                                xb, yb, zb = mg400_server.world_to_base(tcp_world)
                                print(f"\n  ⚠⚠⚠ 碰撞警报 ⚠⚠⚠")
                                print(f"  机械臂TCP已进入障碍物安全范围 (表面距离 {surf_dist*1000:.0f}mm < 50mm)")
                                print(f"  TCP当前位置 (世界): X={tcp_world[0]*1000:.1f} Y={tcp_world[1]*1000:.1f} Z={tcp_world[2]*1000:.1f} mm")
                                print(f"  TCP当前位置 (基座): X={xb:.1f} Y={yb:.1f} Z={zb:.1f} mm")
                                print(f"  障碍物位置 (世界): X={obs_pos[0]*1000:.1f} Y={obs_pos[1]*1000:.1f} Z={obs_pos[2]*1000:.1f} mm")
                                print(f"  运动已暂停 — 请移开障碍物后重新发送目标!")
                                print(f"  ═══════════════════════════════════")
                                ext.collision_info = {
                                    "collided": True,
                                    "surface_dist_mm": surf_dist * 1000,
                                    "tcp_world_mm": [tcp_world[0] * 1000, tcp_world[1] * 1000, tcp_world[2] * 1000],
                                    "tcp_base_mm": [xb, yb, zb],
                                    "obstacle_world_mm": [obs_pos[0] * 1000, obs_pos[1] * 1000, obs_pos[2] * 1000],
                                }
                                ext.active = False
                                ext.target = None
                                ext.waypoints = None
                                ext.wp_idx = -1

                        # 关节限位感知: 处于限位且无法继续同向运动的关节从IK中排除
                        # _grad = error·J_col: _grad>0 → dq应为正; _grad<0 → dq应为负
                        JNT_MARGIN = 0.002  # 缩小限位余量, 最大化可用关节行程
                        repulsion = np.zeros(3)  # 限位排斥偏置 (远离限位)
                        for _ji, _qi in enumerate(IK_JOINTS):
                            _lo, _hi = _effective_joint_range(model, data, _qi)
                            if data.qpos[_qi] <= _lo + JNT_MARGIN:
                                if error @ J[:, _ji] < 0:
                                    J[:, _ji] = 0.0
                                repulsion[_ji] = 0.015  # 远离下限
                            elif data.qpos[_qi] >= _hi - JNT_MARGIN:
                                if error @ J[:, _ji] > 0:
                                    J[:, _ji] = 0.0
                                repulsion[_ji] = -0.015  # 远离上限
                            else:
                                # 软排斥: 接近限位 (< 0.03 rad) 时轻微推开
                                margin_lo = data.qpos[_qi] - _lo
                                margin_hi = _hi - data.qpos[_qi]
                                if margin_lo < 0.03:
                                    repulsion[_ji] = (0.03 - margin_lo) * 0.5
                                elif margin_hi < 0.03:
                                    repulsion[_ji] = -(0.03 - margin_hi) * 0.5

                        # 所有IK关节均被限制 → 强力脱离限位
                        if np.allclose(J, 0.0):
                            for _ji, _qi in enumerate(IK_JOINTS):
                                _lo, _hi = _effective_joint_range(model, data, _qi)
                                _mid = (_lo + _hi) / 2
                                new_val = data.qpos[_qi] + 0.03 * np.sign(_mid - data.qpos[_qi])
                                data.qpos[_qi] = np.clip(new_val, _lo + 0.001, _hi - 0.001)
                            _sync_mimic_joints(model, data)
                            mujoco.mj_forward(model, data)
                            tcp_pos = data.site_xpos[tcp_site_id]
                            error = ik_target - tcp_pos
                            dist = np.linalg.norm(error)
                            mujoco.mj_jacSite(model, data, jac_buf, None, tcp_site_id)
                            J = np.zeros((3, 3))
                            J[:, 0] = jac_buf[:, 0]
                            J[:, 1] = jac_buf[:, 1] - jac_buf[:, 3] + jac_buf[:, 6] - jac_buf[:, 7]
                            J[:, 2] = jac_buf[:, 2] - jac_buf[:, 4] + jac_buf[:, 8]

                        lam = 0.001 + speed * 0.008 if dist < 0.020 else 0.005 + speed * 0.015
                        A = J @ J.T + lam * np.eye(3)

                        max_step = 0.005 + speed * 0.025
                        if dist < 0.005:
                            error = error  # 最终逼近: 不做步长截断
                        elif dist > max_step * 2:
                            error = error / dist * max_step

                        dq = J.T @ np.linalg.solve(A, error) + repulsion

                        # J1 方向校正: 局部梯度可能指向"短路径"(撞限位),
                        # 而全局正确路径需要J1反向旋转(经过0°到目标方位)
                        j1_cur = data.qpos[IK_JOINTS[0]]
                        j1_tgt = np.arctan2(ik_target[1], ik_target[0])
                        j1_lo, j1_hi = model.jnt_range[IK_JOINTS[0]]
                        pos_gap = j1_hi - j1_cur       # 正向还能走多少
                        pos_dist = (j1_tgt - j1_cur + 2*np.pi) % (2*np.pi)  # 正向距离
                        neg_gap = j1_cur - j1_lo       # 负向还能走多少
                        neg_dist = (j1_cur - j1_tgt + 2*np.pi) % (2*np.pi)  # 负向距离
                        if pos_gap >= pos_dist - 0.01:
                            j1_to_tgt = pos_dist
                        elif neg_gap >= neg_dist - 0.01:
                            j1_to_tgt = -neg_dist
                        else:
                            j1_to_tgt = 0.0
                        if abs(j1_to_tgt) > 1.0 and abs(dq[0]) > 1e-8:
                            if np.sign(dq[0]) != np.sign(j1_to_tgt):
                                dq[0] = np.sign(j1_to_tgt) * min(abs(dq[0]), 0.05)

                        max_dq = 0.07 + speed * 0.05  # 最低0.07rad(4°/帧), 保证关节有足够速度
                        dq_clipped = np.clip(dq, -max_dq, max_dq)
                        JNT_MARGIN = 0.002  # 限位安全余量 (rad), 比锁定时略大防止贴死
                        for i, qi in enumerate(IK_JOINTS):
                            new_val = data.qpos[qi] + dq_clipped[i]
                            lo, hi = _effective_joint_range(model, data, qi)
                            data.qpos[qi] = np.clip(new_val, lo + JNT_MARGIN, hi - JNT_MARGIN)

                        _sync_mimic_joints(model, data)

                    # 更新外部状态 + 路径点推进 + 停滞自检
                    if ext_now and ik_target is not None:
                        # 停滞检测 (每帧)
                        if dist < best_dist - 0.0005:
                            best_dist = dist
                            stall_frames = 0
                        else:
                            stall_frames += 1

                        with ext.lock:
                            ext.tcp[:] = tcp_pos
                            ext.error = dist
                            # 计算到最终目标的误差 (供控制面板实时显示)
                            if ext.target is not None:
                                ext.final_error = float(np.linalg.norm(tcp_pos - ext.target))
                            else:
                                ext.final_error = 0.0
                            wp_list = ext.waypoints
                            wp_idx = ext.wp_idx

                            # 路径点变更时重置停滞计数
                            if wp_list is not current_wp_list or wp_idx != current_wp_idx:
                                stall_frames = 0
                                best_dist = 999.0
                                stall_cycles = 0
                                current_wp_list = wp_list
                                current_wp_idx = wp_idx

                            # ── 路径点推进逻辑 ──
                            if wp_list is not None and 0 <= wp_idx < len(wp_list):
                                # 自适应到达阈值: 段长*0.8, 上限20mm, 下限3mm
                                if wp_idx < len(wp_list) - 1:
                                    next_wp_arr = np.array(wp_list[wp_idx + 1])
                                    seg_len = np.linalg.norm(next_wp_arr - wp_list[wp_idx])
                                    arrive_tol = np.clip(seg_len * 0.8, 0.003, 0.020)
                                else:
                                    seg_len = 0.050      # 最终路径点, 回退用默认段长
                                    arrive_tol = 0.002   # 最终目标: 2mm 以内算到达

                                # ③ 越界检测 (主推进机制): TCP 离下一路径点比当前更近 → 立即推进
                                if wp_idx < len(wp_list) - 1:
                                    next_wp = np.array(wp_list[wp_idx + 1])
                                    dist_to_next = np.linalg.norm(tcp_pos - next_wp)
                                    if dist_to_next < dist:
                                        ext.wp_idx = wp_idx + 1
                                        ext.error = dist_to_next
                                        stall_frames = 0; best_dist = 999.0
                                        current_wp_idx = ext.wp_idx
                                        continue  # 跳过其余条件, 下帧再判

                                # ① 到达阈值推进 (备用): 误差 < 自适应阈值时推进
                                if dist < arrive_tol and wp_idx < len(wp_list) - 1:
                                    ext.wp_idx = wp_idx + 1
                                    stall_frames = 0; best_dist = 999.0
                                    current_wp_idx = ext.wp_idx
                                    if wp_idx % 10 == 0:
                                        print(f"  路径点 {ext.wp_idx}/{len(wp_list)}, err={dist*1000:.1f}mm")
                                elif dist < arrive_tol:
                                    # ② 到达最终路径点 → 目标完成
                                    ext.active = False
                                    ext.target = None
                                    ext.waypoints = None
                                    ext.wp_idx = -1
                                    ext.reached = True
                                    ext.error = dist
                                    print(f"  目标到达: TCP={tcp_pos}, err={dist*1000:.1f}mm")
                                    stall_frames = 0; best_dist = 999.0

                                # ④ 停滞 > 1.5s (≈45帧) 且未到达: 强制推进
                                if stall_frames > 45 and dist >= arrive_tol:
                                    if wp_idx < len(wp_list) - 1:
                                        ext.wp_idx = wp_idx + 1
                                        stall_frames = 0; best_dist = 999.0
                                        current_wp_idx = ext.wp_idx
                                        print(f"  停滞 → 强制路径点 {ext.wp_idx}/{len(wp_list)} (err={dist*1000:.0f}mm)")
                                    else:
                                        if dist < 0.005:
                                            ext.active = False
                                            ext.target = None
                                            ext.waypoints = None
                                            ext.wp_idx = -1
                                            ext.reached = True
                                            ext.error = dist
                                            print(f"  目标近似到达 (停滞, err={dist*1000:.1f}mm)")
                                            stall_frames = 0; best_dist = 999.0
                                            stall_cycles = 0
                                        else:
                                            stall_cycles += 1
                                            if stall_cycles == 1:
                                                # 诊断: 打印当前关节角度和限位状态
                                                print(f"  ┌─ 停滞诊断 ─────────────────────────────")
                                                print(f"  │ TCP位置: [{tcp_pos[0]:.4f} {tcp_pos[1]:.4f} {tcp_pos[2]:.4f}]")
                                                print(f"  │ 目标: {ext.target}")
                                                for _qi in IK_JOINTS:
                                                    _lo, _hi = model.jnt_range[_qi]
                                                    _at_lo = data.qpos[_qi] <= _lo + 0.003
                                                    _at_hi = data.qpos[_qi] >= _hi - 0.003
                                                    _flag = " <<限位" if _at_lo or _at_hi else ""
                                                    print(f"  │ J{_qi+1}: {data.qpos[_qi]:.4f} rad ({np.degrees(data.qpos[_qi]):.1f}°)  range=[{_lo:.4f},{_hi:.4f}]{_flag}")
                                                print(f"  └────────────────────────────────────────")
                                            if stall_cycles >= 2:
                                                replan_count += 1
                                                if replan_count >= 2:
                                                    ext.active = False
                                                    ext.target = None
                                                    ext.waypoints = None
                                                    ext.wp_idx = -1
                                                    ext.error = dist
                                                    print(f"  目标不可达 (err={dist*1000:.0f}mm, {replan_count}次重算后放弃)")
                                                    stall_frames = 0; best_dist = 999.0
                                                    stall_cycles = 0
                                                    replan_count = 0
                                                else:
                                                    new_wps = _compute_waypoints(tcp_pos.tolist(), ext.target)
                                                    if new_wps:
                                                        ext.waypoints = new_wps
                                                        ext.wp_idx = 0
                                                        stall_frames = 0; best_dist = 999.0
                                                        stall_cycles = 0
                                                        current_wp_list = new_wps
                                                        current_wp_idx = 0
                                                        print(f"  重算路径 [{replan_count}/2]: 从当前位置到目标, {len(new_wps)}段")
                                                    else:
                                                        ext.active = False
                                                        ext.target = None
                                                        ext.waypoints = None
                                                        ext.wp_idx = -1
                                                        ext.error = dist
                                                        print(f"  目标不可达 (err={dist*1000:.0f}mm, 重算失败)")
                                                        stall_frames = 0; best_dist = 999.0
                                                        stall_cycles = 0
                                                        replan_count = 0
                                            else:
                                                if unreachable_cooldown <= 0:
                                                    print(f"  目标暂不可达 (err={dist*1000:.0f}mm), 继续尝试 ({stall_cycles}/2)...")
                                                    unreachable_cooldown = 600
                                                else:
                                                    unreachable_cooldown -= 1
                                                stall_frames = 0
                                                best_dist = dist

                # ── 方向键 (仅自由拖拽模式, 且障碍物未激活) ──
                if free_drag[0] and not ext.obstacle_active:
                    step = 0.03
                    sj = selected_joint[0]
                    updated = False
                    if _user32.GetAsyncKeyState(0x26) & 0x8000:
                        data.qpos[ACTUATED_QPOS[sj]] += step
                        updated = True
                    if _user32.GetAsyncKeyState(0x28) & 0x8000:
                        data.qpos[ACTUATED_QPOS[sj]] -= step
                        updated = True
                    if updated:
                        JNT_MARGIN = 0.002
                        for qi in ACTUATED_QPOS:
                            lo, hi = _effective_joint_range(model, data, qi)
                            data.qpos[qi] = np.clip(data.qpos[qi], lo + JNT_MARGIN, hi - JNT_MARGIN)
                        _sync_mimic_joints(model, data)
                        mujoco.mj_forward(model, data)
                        drag["target"][:] = data.site_xpos[tcp_site_id]

                # ── Ctrl = qpos ──
                for i, qidx in enumerate(ACTUATED_QPOS):
                    data.ctrl[i] = data.qpos[qidx]

                # ── 重力补偿 + 速度阻尼 ──
                qvel_save = data.qvel.copy()
                qacc_save = data.qacc.copy()
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
                mujoco.mj_rne(model, data, 0, grav_comp)
                data.qvel[:] = qvel_save
                data.qacc[:] = qacc_save
                data.qfrc_applied[:] = grav_comp - 0.2 * data.qvel  # 降低速度阻尼, 减小伺服静差

            # ── 障碍物移动 + mocap 同步 (边沿触发, 每次按键移动 20mm) ──
            if ext.obstacle_active:
                obs_step = 0.020  # 20mm/次
                with ext.lock:
                    # 边沿检测: 只在按下瞬间 (0→1) 触发一次
                    for vk, delta in _OBS_KEY_MAP:
                        pressed = bool(_user32.GetAsyncKeyState(vk) & 0x8000)
                        if pressed and not _obs_key_prev[vk]:
                            ext.obstacle_pos[0] += delta[0] * obs_step
                            ext.obstacle_pos[1] += delta[1] * obs_step
                            ext.obstacle_pos[2] += delta[2] * obs_step
                        _obs_key_prev[vk] = pressed
                # 同步 mocap 位置以更新渲染
                if obs_mocap_id >= 0:
                    data.mocap_pos[obs_mocap_id] = ext.obstacle_pos.copy()

            # ── 每帧更新 TCP 位置 (供外部查询) ──
            with ext.lock:
                ext.tcp[:] = data.site_xpos[tcp_site_id]
                cur_target = ext.target if ext.active else None

            # ── 更新目标点标记 (mocap) ──
            if target_marker_mocap_id >= 0 and cur_target is not None:
                data.mocap_pos[target_marker_mocap_id] = cur_target

            # ── 动态更新 J3 执行器 ctrlrange (J2-J3 耦合: 仅上限) ──
            JNT_MARGIN = 0.005
            j2 = data.qpos[1]
            j3_lo = model.jnt_range[2, 0]
            if j2 < 0.0:
                j2_norm = np.clip((j2 + 0.4363) / 0.4363, 0.0, 1.0)
                j3_hi = 1.0 + 0.833 * j2_norm           # J2=-25°→1.0,  J2=0°→1.833
            else:
                j3_hi = model.jnt_range[2, 1]
            model.actuator_ctrlrange[2, 0] = j3_lo + JNT_MARGIN
            model.actuator_ctrlrange[2, 1] = j3_hi - JNT_MARGIN
            # 同时裁剪当前 ctrl 以免超出新范围
            data.ctrl[2] = np.clip(data.ctrl[2], model.actuator_ctrlrange[2, 0], model.actuator_ctrlrange[2, 1])

            was_soft = soft_mode
            mujoco.mj_step(model, data)
            _sync_mimic_joints(model, data)  # 渲染前修正 mimic 关节, 防止脱节
            viewer.sync()


if __name__ == "__main__":
    main()
