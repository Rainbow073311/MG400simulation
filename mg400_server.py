"""
MG400 Dashboard 协议服务器 — 模拟真实 MG400 机器人的 TCP 接口。
- 端口 29999: Dashboard 文本命令 (控制)
- 端口 30004: 反馈 1440 字节二进制包 (状态流)

与 simulate_slider.py 通过 ext 命名空间共享状态。
"""
import json
import math
import os
import re
import socket
import sys
import threading
import time

import numpy as np

# 复用官方库的 1440 字节反馈包结构
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TCP-IP-Python-V4"))
from dobot_api import MyType

BASE_ROT = math.radians(45)
COS45 = math.cos(BASE_ROT)
SIN45 = math.sin(BASE_ROT)


# ── 坐标转换 ──

def world_to_base(world_pos):
    """世界坐标 (m) → Dobot 基座坐标 (mm)。基座 X+ 指向世界 45°。"""
    x_w, y_w, z_w = world_pos
    x_b = x_w * COS45 + y_w * SIN45
    y_b = -x_w * SIN45 + y_w * COS45
    return x_b * 1000.0, y_b * 1000.0, z_w * 1000.0


def base_to_world(x_mm, y_mm, z_mm):
    """Dobot 基座坐标 (mm) → 世界坐标 (m)。"""
    x_b, y_b, z_b = x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0
    x_w = x_b * COS45 - y_b * SIN45
    y_w = x_b * SIN45 + y_b * COS45
    return x_w, y_w, z_b


def _rotmat_to_euler_xyz(R_flat):
    """MuJoCo 3x3 旋转矩阵 (行主序) → Rx,Ry,Rz (度)。"""
    r00, r01, r02 = R_flat[0], R_flat[1], R_flat[2]
    r10, r11, r12 = R_flat[3], R_flat[4], R_flat[5]
    r20, r21, r22 = R_flat[6], R_flat[7], R_flat[8]
    if abs(r20) < 0.999999:
        ry = math.asin(-r20)
        rx = math.atan2(r21, r22)
        rz = math.atan2(r10, r00)
    else:
        rx = 0.0
        ry = math.copysign(math.pi / 2, -r20)
        rz = math.atan2(-r01, r11)
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


# ── Dashboard 命令解析 ──

def _split_args(s):
    """按逗号分割, 尊重嵌套 {}。"""
    parts = []
    depth = 0
    cur = []
    for ch in s:
        if ch == '{':
            depth += 1
            cur.append(ch)
        elif ch == '}':
            depth -= 1
            cur.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


def _parse_val(s):
    """解析单个值: 数字, {a,b,c} 列表, 或字符串。"""
    s = s.strip()
    if s.startswith('{') and s.endswith('}'):
        inner = s[1:-1]
        return [] if not inner else [float(x.strip()) for x in inner.split(',')]
    try:
        if '.' in s or 'e' in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_dashboard_cmd(text):
    """解析 'FuncName(arg1,kw=val)' → (name, args_list, kwargs_dict)。"""
    text = text.strip()
    m = re.match(r'(\w+)\((.*)\)', text, re.DOTALL)
    if not m:
        return None, [], {}
    name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return name, [], {}
    args = []
    kwargs = {}
    for part in _split_args(args_str):
        if not part:
            continue
        if '=' in part:
            key, val = part.split('=', 1)
            kwargs[key.strip()] = _parse_val(val.strip())
        else:
            args.append(_parse_val(part))
    return name, args, kwargs


# ── 反馈包构造 ──

def make_feedback_packet(robot_mode, speed_scaling, q_actual_deg, tcp_base_mm,
                         command_id=0, digital_inputs=0, digital_outputs=0):
    """构造 1440 字节反馈包, 返回 numpy 数组 (MyType 结构)。"""
    buf = np.zeros(1, dtype=MyType)
    pkt = buf[0]
    pkt["len"] = 1440
    pkt["TestValue"] = 0x123456789ABCDEF
    pkt["RobotMode"] = int(robot_mode)
    pkt["SpeedScaling"] = float(speed_scaling)
    pkt["DigitalInputs"] = int(digital_inputs)
    pkt["DigitalOutputs"] = int(digital_outputs)
    pkt["CurrentCommandId"] = int(command_id)
    for i in range(min(6, len(q_actual_deg))):
        pkt["QActual"][i] = float(q_actual_deg[i])
        pkt["QTarget"][i] = float(q_actual_deg[i])
    for i in range(6):
        pkt["ToolVectorActual"][i] = float(tcp_base_mm[i])
        pkt["ToolVectorTarget"][i] = float(tcp_base_mm[i])
    return buf


# ── Dashboard 命令处理 ──

class DashboardHandler:
    """接收 Dashboard 文本命令, 通过 ext 命名空间驱动仿真。"""

    def __init__(self, state):
        self.s = state       # model, data, tcp_site_id, Z_INITIAL, _HOME_TCP, 函数引用
        self.e = state["ext"]  # ext 命名空间: lock, active, target, waypoints, ...
        self._cmd_id = 0

    def _next_id(self):
        self._cmd_id += 1
        return self._cmd_id

    def handle(self, text):
        name, args, kwargs = parse_dashboard_cmd(text)
        if name is None:
            return "-1"
        method = getattr(self, f"_cmd_{name}", None)
        if method is None:
            return "-1"
        try:
            return method(args, kwargs)
        except Exception:
            return "-1"

    # ── 生命周期 ──

    def _cmd_EnableRobot(self, args, kwargs):
        with self.e.lock:
            self.e.active = False
            self.e.target = None
            self.e.waypoints = None
            self.e.wp_idx = -1
            self.e.reached = True
        return "0"

    def _cmd_DisableRobot(self, args, kwargs):
        return self._cmd_EnableRobot(args, kwargs)

    def _cmd_ClearError(self, args, kwargs):
        return "0"

    def _cmd_Stop(self, args, kwargs):
        with self.e.lock:
            self.e.active = False
            self.e.target = None
            self.e.waypoints = None
            self.e.wp_idx = -1
        return "0"

    def _cmd_PowerOn(self, args, kwargs):
        return "0"

    # ── 速度 ──

    def _cmd_SpeedFactor(self, args, kwargs):
        ratio = args[0] if args else 30
        speed = max(0.05, min(1.0, float(ratio) / 100.0))
        with self.e.lock:
            self.e.speed = speed
        return "0"

    # ── 查询 ──

    def _cmd_RobotMode(self, args, kwargs):
        with self.e.lock:
            active = self.e.active
        return "7" if active else "5"

    def _cmd_GetAngle(self, args, kwargs):
        data = self.s["data"]
        j = [math.degrees(data.qpos[i]) for i in range(4)] + [0.0, 0.0]
        return "{" + ",".join(f"{v:.4f}" for v in j) + "}"

    def _cmd_GetPose(self, args, kwargs):
        data = self.s["data"]
        tcp_world = data.site_xpos[self.s["tcp_site_id"]].copy()
        x_mm, y_mm, z_mm = world_to_base(tcp_world)
        xmat = data.site_xmat[self.s["tcp_site_id"]].copy()
        rx, ry, rz = _rotmat_to_euler_xyz(xmat)
        return "{" + ",".join(f"{v:.4f}" for v in [x_mm, y_mm, z_mm, rx, ry, rz]) + "}"

    def _cmd_GetErrorID(self, args, kwargs):
        return "{0}"

    # ── I/O (仿真返回固定值) ──

    def _cmd_DO(self, args, kwargs):       return "0"
    def _cmd_DOInstant(self, args, kwargs): return "0"
    def _cmd_GetDO(self, args, kwargs):    return "0"
    def _cmd_DI(self, args, kwargs):       return "0"
    def _cmd_ToolDO(self, args, kwargs):   return "0"
    def _cmd_ToolDI(self, args, kwargs):   return "0"
    def _cmd_AO(self, args, kwargs):       return "0"
    def _cmd_AI(self, args, kwargs):       return "0.0"

    # ── 障碍物控制 ──

    def _cmd_SetObstacle(self, args, kwargs):
        """SetObstacle(active=1) 或 SetObstacle(pos={x,y,z})"""
        active = kwargs.get("active")
        pos = kwargs.get("pos")
        with self.e.lock:
            if active is not None:
                self.e.obstacle_active = bool(int(active))
            if pos is not None and isinstance(pos, list) and len(pos) >= 3:
                self.e.obstacle_pos[:] = [float(pos[0]), float(pos[1]), float(pos[2])]
            active_str = "1" if self.e.obstacle_active else "0"
            pos_str = f"{{{self.e.obstacle_pos[0]:.3f},{self.e.obstacle_pos[1]:.3f},{self.e.obstacle_pos[2]:.3f}}}"
        return f"0,{{active={active_str},pos={pos_str}}}"

    def _cmd_GetObstacle(self, args, kwargs):
        with self.e.lock:
            active = self.e.obstacle_active
            pos = self.e.obstacle_pos.copy()
        return f"{{active={1 if active else 0},pos={{{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}}}}}"

    def _cmd_GetCollisionStatus(self, args, kwargs):
        """查询最近一次碰撞信息。无碰撞时返回 {collided=0}。"""
        info = self.e.collision_info
        if info is None:
            return "{collided=0}"
        return (f"{{collided=1,"
                f"surf_dist_mm={info['surface_dist_mm']:.1f},"
                f"tcp_world_mm={{{info['tcp_world_mm'][0]:.1f},{info['tcp_world_mm'][1]:.1f},{info['tcp_world_mm'][2]:.1f}}},"
                f"tcp_base_mm={{{info['tcp_base_mm'][0]:.1f},{info['tcp_base_mm'][1]:.1f},{info['tcp_base_mm'][2]:.1f}}},"
                f"obstacle_world_mm={{{info['obstacle_world_mm'][0]:.1f},{info['obstacle_world_mm'][1]:.1f},{info['obstacle_world_mm'][2]:.1f}}}}}")

    # ── 运动 ──

    def _cmd_MovJ(self, args, kwargs):
        """MovJ(pose={x,y,z,rx,ry,rz}) 或 MovJ(joint={j1,j2,j3,j4,j5,j6})"""
        pose = kwargs.get("pose")
        joint = kwargs.get("joint")
        if pose is not None:
            values = pose if isinstance(pose, list) else []
            return self._move_pose(values)
        elif joint is not None:
            values = joint if isinstance(joint, list) else []
            return self._move_joint(values)
        return "-1"

    def _cmd_MovL(self, args, kwargs):
        return self._cmd_MovJ(args, kwargs)

    def _cmd_MoveJog(self, args, kwargs):
        return "0"  # 仿真暂不支持点动

    def _move_pose(self, values):
        """位姿模式: 基座坐标 mm → 世界坐标 m, 规划安全路径。"""
        if len(values) < 3:
            return "-1"
        x_mm, y_mm, z_mm = float(values[0]), float(values[1]), float(values[2])
        x_w, y_w, z_w = base_to_world(x_mm, y_mm, z_mm)

        # J1 方位角校验
        theta_world = math.degrees(math.atan2(y_w, x_w))
        j1_needed = (theta_world - 45 + 180) % 360 - 180
        if abs(j1_needed) > 160:
            return "-1"

        with self.e.lock:
            cur = self.e.tcp.tolist()
            speed = self.e.speed
            wp_list = self.s["_plan_safe_motion"](cur, [x_w, y_w, z_w], verbose_prefix="")
            z_init = self.s["Z_INITIAL"]
            unreachable = False
            all_pts = [cur] + wp_list
            for i, wp in enumerate(wp_list):
                if self.s["_in_unreachable_zone"](wp, z_init):
                    unreachable = True
                    break
                if self.s["_segment_crosses_unreachable"](all_pts[i], wp, z_init):
                    unreachable = True
                    break

            if unreachable:
                self.e.waypoints = None; self.e.wp_idx = -1
                self.e.active = False; self.e.reached = False
                return "-1"

            self.e.speed = speed
            self.e.target = [x_w, y_w, z_w]
            self.e.waypoints = wp_list if wp_list else None
            self.e.wp_idx = 0 if self.e.waypoints else -1
            self.e.active = True
            self.e.reached = False
            self.e.collision_info = None

        cid = self._next_id()
        return f"0,{{{cid}}}"

    def _move_joint(self, values):
        """关节模式: 标记 joint_move, 由主循环计算 FK 后规划路径。"""
        if len(values) < 4:
            return "-1"
        j_deg = [float(values[i]) if i < len(values) else 0.0 for i in range(6)]
        j_rad = [math.radians(v) for v in j_deg[:4]]

        with self.e.lock:
            self.e.target_joints = j_rad
            self.e.joint_move = True
            self.e.active = True
            self.e.reached = False
            self.e.collision_info = None

        cid = self._next_id()
        return f"0,{{{cid}}}"


# ── TCP 服务器 ──

def run_dashboard_server(state):
    """Dashboard 协议服务器 — 端口 29999。"""
    handler = DashboardHandler(state)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 29999))
    server.listen(1)
    server.settimeout(1.0)
    print("  Dashboard server: port 29999 (Dobot 协议)")

    conn = None
    buf = ""
    while True:
        try:
            if conn is None:
                try:
                    conn, addr = server.accept()
                    print(f"  Dashboard 客户端已连接: {addr}")
                except socket.timeout:
                    continue
            data = conn.recv(4096)
            if not data:
                conn.close(); conn = None
                print("  Dashboard 客户端已断开")
                continue
            buf += data.decode("utf-8")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                resp = handler.handle(line)
                conn.sendall((resp + "\n").encode("utf-8"))
        except (ConnectionResetError, BrokenPipeError, OSError):
            if conn:
                conn.close(); conn = None
                print("  Dashboard 客户端已断开")


def run_feedback_server(state):
    """反馈流服务器 — 端口 30004, ~30 Hz 推送 1440 字节二进制包。"""
    e = state["ext"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 30004))
    server.listen(1)
    server.settimeout(1.0)
    print("  Feedback server: port 30004 (1440 字节流)")

    conn = None
    while True:
        try:
            if conn is None:
                try:
                    conn, addr = server.accept()
                    conn.setblocking(False)
                    print(f"  Feedback 客户端已连接: {addr}")
                except socket.timeout:
                    continue

            with e.lock:
                data = state["data"]
                tcp_world = data.site_xpos[state["tcp_site_id"]].copy()
                xmat = data.site_xmat[state["tcp_site_id"]].copy()
                q = [math.degrees(data.qpos[i]) for i in [0, 1, 2, 5]] + [0.0, 0.0]
                active = e.active
                speed = e.speed * 100.0

            x_mm, y_mm, z_mm = world_to_base(tcp_world)
            rx, ry, rz = _rotmat_to_euler_xyz(xmat)
            robot_mode = 7 if active else 5

            pkt = make_feedback_packet(
                robot_mode=robot_mode, speed_scaling=speed,
                q_actual_deg=q,
                tcp_base_mm=[x_mm, y_mm, z_mm, rx, ry, rz],
            )
            try:
                conn.sendall(pkt.tobytes())
            except (ConnectionResetError, BrokenPipeError, OSError):
                conn.close(); conn = None
                print("  Feedback 客户端已断开")
            time.sleep(0.03)
        except (ConnectionResetError, BrokenPipeError, OSError):
            if conn:
                conn.close(); conn = None
                print("  Feedback 客户端已断开")


def start_servers(state):
    """启动 Dashboard + Feedback 服务器线程。"""
    threading.Thread(target=run_dashboard_server, args=(state,), daemon=True).start()
    threading.Thread(target=run_feedback_server, args=(state,), daemon=True).start()
