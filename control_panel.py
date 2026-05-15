"""
MG400 控制面板 — 独立运行的图形化交互脚本
使用 Dobot Dashboard 协议 (端口 29999) 与仿真/真实 MG400 通信。

用法:
    python control_panel.py
    python control_panel.py --host localhost --port 29999
"""
import argparse
import json
import re
import socket
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np


class MG400ControlPanel:
    """控制面板: Dobot Dashboard 协议客户端 + tkinter GUI"""

    def __init__(self, host="localhost", port=29999, ext_port=9878):
        self.host = host
        self.port = port
        self.ext_port = ext_port
        self.sock = None
        self._reconnect_cooldown = 0
        self._speed_after_id = None
        self._z_initial_mm = 120.0   # TCP Z 基准 (mm, 基座坐标), 首次 GetPose 后更新
        self._ext_auto_send = True  # 外部输入是否自动发送
        self._ext_server_sock = None

        # ── 主窗口 ──
        self.window = tk.Tk()
        self.window.title("MG400 控制面板")
        self.window.geometry("540x520")
        self.window.resizable(True, True)
        self.window.minsize(420, 480)
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.attributes("-topmost", True)  # 始终浮于 MuJoCo 窗口上方

        # 键盘快捷键
        def _skip_if_editing(e):
            w = e.widget
            return isinstance(w, (tk.Entry, tk.Text, ttk.Entry))
        self.window.bind("<Return>", lambda e: self._send_target())  # Enter 始终发送
        self.window.bind("<space>", lambda e: self._stop() if not _skip_if_editing(e) else None)
        self.window.bind("<h>", lambda e: self._go_home() if not _skip_if_editing(e) else None)

        def _focus_cycle(direction):
            """在 X→Y→Z 输入框之间循环切换焦点。direction: +1 向下, -1 向上。"""
            entries = [self.x_entry, self.y_entry, self.z_entry]
            focused = self.window.focus_get()
            try:
                idx = entries.index(focused)
            except ValueError:
                idx = -1 if direction > 0 else 0
            nxt = (idx + direction) % 3
            entries[nxt].focus_set()
            entries[nxt].selection_range(0, "end")
        self.window.bind("<Up>", lambda e: _focus_cycle(-1))
        self.window.bind("<Down>", lambda e: _focus_cycle(+1))

        # 样式
        style = ttk.Style()
        style.theme_use("clam")

        # ── 连接状态 ──
        conn_frame = ttk.LabelFrame(self.window, text="连接", padding=10)
        conn_frame.pack(fill="x", padx=12, pady=(12, 0))

        self.conn_label = ttk.Label(conn_frame, text="未连接", foreground="red")
        self.conn_label.pack(side="left")
        ttk.Button(conn_frame, text="连接", command=self._connect, width=8).pack(side="right")

        # ── 实时位置 ──
        pos_frame = ttk.LabelFrame(self.window, text="末端执行器实时位置 (mm)", padding=10)
        pos_frame.pack(fill="x", padx=12, pady=8)

        row1 = ttk.Frame(pos_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="X:", width=3).pack(side="left")
        self.x_pos_label = ttk.Label(row1, text="---", width=8, relief="sunken")
        self.x_pos_label.pack(side="left", padx=(0, 8))
        ttk.Label(row1, text="Y:", width=3).pack(side="left")
        self.y_pos_label = ttk.Label(row1, text="---", width=8, relief="sunken")
        self.y_pos_label.pack(side="left", padx=(0, 8))
        ttk.Label(row1, text="Z:", width=3).pack(side="left")
        self.z_pos_label = ttk.Label(row1, text="---", width=8, relief="sunken")
        self.z_pos_label.pack(side="left")

        row2 = ttk.Frame(pos_frame)
        row2.pack(fill="x", pady=(4, 0))
        self.error_label = ttk.Label(row2, text="误差: --- m")
        self.error_label.pack(side="left")
        self.reached_label = ttk.Label(row2, text="")
        self.reached_label.pack(side="right")

        # ── 目标输入 ──
        target_frame = ttk.LabelFrame(self.window, text="目标坐标 (Dobot 基座坐标, X/Y绝对 Z相对, 毫米)", padding=10)
        target_frame.pack(fill="x", padx=12, pady=8)


        def make_input(parent, label, initial):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=4).pack(side="left")
            var = tk.StringVar(value=initial)
            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side="left")
            hint = {"X:": "mm 基座坐标", "Y:": "mm 基座坐标", "Z:": "mm 相对TCP原点"}.get(label, "")
            ttk.Label(row, text=hint, width=16, foreground="gray").pack(side="left")
            return var, entry

        self.x_var, self.x_entry = make_input(target_frame, "X:", "300")
        self.y_var, self.y_entry = make_input(target_frame, "Y:", "0")
        self.z_var, self.z_entry = make_input(target_frame, "Z:", "0")

        # 验证提示标签 (醒目, 位于输入框与按钮之间)
        self.validation_label = tk.Label(
            target_frame, text="", fg="red", font=("Microsoft YaHei", 9, "bold"),
            anchor="w", justify="left", wraplength=490
        )
        self.validation_label.pack(fill="x", pady=(4, 0))

        # 输入框内容变化时清除验证提示
        for var in (self.x_var, self.y_var, self.z_var):
            var.trace_add("write", lambda *_: self._clear_validation())

        btn_row = ttk.Frame(target_frame)
        btn_row.pack(fill="x", pady=(8, 0))
        self.move_btn = ttk.Button(btn_row, text="▶  移动到目标", command=self._send_target)
        self.move_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.stop_btn = ttk.Button(btn_row, text="■ 停止", command=self._stop)
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # ── 速度调节 ──
        speed_frame = ttk.LabelFrame(self.window, text="移动速度", padding=10)
        speed_frame.pack(fill="x", padx=12, pady=8)

        speed_row = ttk.Frame(speed_frame)
        speed_row.pack(fill="x")
        ttk.Label(speed_row, text="慢").pack(side="left")
        self.speed_var = tk.DoubleVar(value=0.3)
        self.speed_slider = ttk.Scale(
            speed_row, from_=0.05, to=1.0, variable=self.speed_var,
            command=self._on_speed_change, length=360
        )
        self.speed_slider.pack(side="left", padx=6, fill="x", expand=True)
        ttk.Label(speed_row, text="快").pack(side="left")
        self.speed_label = ttk.Label(speed_row, text="30%")
        self.speed_label.pack(side="right")

        # ── 回零按钮 ──
        home_frame = ttk.Frame(self.window)
        home_frame.pack(fill="x", padx=12, pady=(0, 4))
        ttk.Button(
            home_frame, text="⟲ 回零 (安全姿态)", command=self._go_home
        ).pack(fill="x")

        # ── 障碍物避障 ──
        obs_frame = ttk.LabelFrame(self.window, text="障碍物避障 (操作员手臂)", padding=8)
        obs_frame.pack(fill="x", padx=12, pady=(0, 4))
        obs_row = ttk.Frame(obs_frame)
        obs_row.pack(fill="x")
        self.obs_btn = ttk.Button(obs_row, text="● 激活障碍物", command=self._toggle_obstacle)
        self.obs_btn.pack(side="left", padx=(0, 8))
        self.obs_status_label = ttk.Label(obs_row, text="已关闭", foreground="gray")
        self.obs_status_label.pack(side="left")
        self.obs_pos_label = ttk.Label(obs_row, text="")
        self.obs_pos_label.pack(side="right")
        self._obs_active = False

        # ── 日志 ──
        log_frame = ttk.LabelFrame(self.window, text="日志", padding=5)
        log_frame.pack(fill="both", expand=True, padx=12, pady=8)
        self.log_text = tk.Text(log_frame, height=6, state="disabled", font=("Consolas", 9))
        self.log_text.pack(fill="both", expand=True)
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # ── 定时器 ──
        self._poll_position()
        self._log("控制面板已启动, 请点击「连接」按钮")

        # ── 外部接口 (大模型坐标输入) ──
        self._start_ext_server()

    # ── 网络通信 (Dobot Dashboard 协议) ──

    def _send_cmd(self, text_cmd):
        """发送文本命令 (Dashboard 协议), 返回响应字符串。连接失败返回 None。"""
        if self.sock is None:
            return None
        try:
            msg = (text_cmd + "\n").encode("utf-8")
            self.sock.sendall(msg)
            resp = self.sock.recv(4096)
            if not resp:
                self._disconnect()
                return None
            return resp.decode("utf-8").strip()
        except (ConnectionResetError, BrokenPipeError, OSError, socket.timeout):
            self._disconnect()
            return None

    def _parse_pose_resp(self, resp):
        """解析 GetPose 响应 '{x,y,z,rx,ry,rz}' → [x,y,z,rx,ry,rz] (float)."""
        if resp is None:
            return None
        try:
            m = re.match(r'\{([-0-9.,eE]+)\}', resp)
            if m:
                return [float(v.strip()) for v in m.group(1).split(',')]
        except (ValueError, AttributeError):
            pass
        return None

    def _send_dashboard_cmd(self, text_cmd):
        """发送 Dashboard 命令, 检查响应是否为成功 (不以 -1 开头)。"""
        resp = self._send_cmd(text_cmd)
        if resp is None:
            return False, None
        # 成功响应: "0", "0,{id}", "{...}" (查询)
        if resp.startswith("-1") and not resp.startswith("-1."):
            return False, resp
        return True, resp

    def _connect(self):
        """连接仿真/真实 MG400 机器人 (Dashboard 端口)"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.5)
            # 使能机器人
            self._send_cmd("EnableRobot()")
            # 获取当前位姿, 确定 Z_INITIAL
            resp = self._send_cmd("GetPose()")
            pose = self._parse_pose_resp(resp)
            if pose and len(pose) >= 3:
                self._z_initial_mm = pose[2]  # 基座坐标 Z (mm)
                self._log(f"当前 TCP Z 基准: {self._z_initial_mm:.0f}mm")
            # 同步速度
            pct = int(float(self.speed_var.get()) * 100)
            self._send_cmd(f"SpeedFactor({pct})")
            self.conn_label.config(text="● 已连接", foreground="green")
            self.move_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self._log(f"已连接到 {self.host}:{self.port}")
            self.validation_label.config(text="✓ 已连接, 可输入坐标", fg="green")
        except (ConnectionRefusedError, socket.timeout, OSError) as e:
            self.conn_label.config(text="未连接 (仿真未启动?)", foreground="red")
            self.sock = None
            self._log(f"连接失败: {e}")

    def _disconnect(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.conn_label.config(text="未连接", foreground="red")
        self.move_btn.config(state="disabled")
        self.stop_btn.config(state="disabled")
        self.x_pos_label.config(text="---")
        self.y_pos_label.config(text="---")
        self.z_pos_label.config(text="---")
        self.error_label.config(text="误差: --- mm")
        self.reached_label.config(text="")
        self._reconnect_cooldown = 0  # 触发自动重连

    def _on_close(self):
        self._disconnect()
        if self._ext_server_sock:
            try:
                self._ext_server_sock.close()
            except OSError:
                pass
        self.window.destroy()

    # ── 命令 ──

    def _check_workspace(self, x_mm, y_mm, z_mm):
        """J1 方位角验证 (Dobot 基座坐标, 方位角 = J1 角度, ±160°)。"""
        theta_base = np.degrees(np.arctan2(y_mm, x_mm))
        if abs(theta_base) > 160:
            return False, f"方位角 θ={theta_base:.0f}° 超出J1范围 (±160°)"
        return True, ""

    def _clear_validation(self):
        """输入框内容变化时清除验证提示"""
        self.validation_label.config(text="", fg="red")

    def _send_target(self):
        """发送目标坐标 (Dobot 基座坐标 mm, MovJ pose)"""
        if self.sock is None:
            self._log("错误: 未连接到仿真")
            self.validation_label.config(text="⚠ 未连接到仿真, 请先点击「连接」", fg="red")
            return
        try:
            x_mm = float(self.x_var.get())
            y_mm = float(self.y_var.get())
            z_rel = float(self.z_var.get())
        except ValueError:
            self._log("错误: 坐标值必须是数字")
            self.validation_label.config(text="⚠ 坐标值必须是数字", fg="red")
            return

        # Z 基准: 用户输入的是相对 Z, 转绝对坐标
        z_mm = self._z_initial_mm + z_rel

        # 工作空间验证
        ok, msg = self._check_workspace(x_mm, y_mm, z_mm)
        if not ok:
            self._log(f"✗ 超出工作范围: {msg}")
            self.validation_label.config(text=f"⚠ 超出工作范围 — {msg}", fg="red")
            return

        self._log(f"✓ 坐标在工作范围内, 发送目标...")
        self.validation_label.config(text="✓ 坐标有效, 正在发送...", fg="green")

        cmd = f"MovJ(pose={{{x_mm:.1f},{y_mm:.1f},{z_mm:.1f},0,0,0}})"
        ok, resp = self._send_dashboard_cmd(cmd)
        if resp is None:
            self._log("错误: 发送失败, 请重新连接")
            self.validation_label.config(text="⚠ 发送失败, 请重新连接", fg="red")
            return
        if ok:
            self._log(f"目标已发送: X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f}mm")
            self.validation_label.config(text=f"✓ 目标已发送 X={x_mm:.0f} Y={y_mm:.0f} Z={z_rel:.0f}mm(相对)", fg="green")
        else:
            self._log(f"错误: {resp}")
            self.validation_label.config(text=f"⚠ 服务器拒绝 — {resp}", fg="red")

    def _on_speed_change(self, val):
        """速度滑块变化时发送到仿真 (防抖: 停止拖动150ms后才发送)"""
        pct = int(float(val) * 100)
        self.speed_label.config(text=f"{pct}%")
        if self._speed_after_id is not None:
            self.window.after_cancel(self._speed_after_id)
        self._speed_after_id = self.window.after(
            150, lambda: self._send_cmd(f"SpeedFactor({int(self.speed_var.get() * 100)})"))

    def _stop(self):
        """停止当前运动"""
        resp = self._send_cmd("Stop()")
        if resp:
            self._log("已发送停止命令")

    def _go_home(self):
        """回零 — 发送 MovJ joint 全零 → 仿真计算 FK 后规划安全路径"""
        if self.sock is None:
            self._log("错误: 未连接到仿真")
            self.validation_label.config(text="⚠ 未连接到仿真", fg="red")
            return
        ok, resp = self._send_dashboard_cmd("MovJ(joint={0,0,0,0,0,0})")
        if resp is None:
            self._log("错误: 发送失败, 请重新连接")
            self.validation_label.config(text="⚠ 发送失败", fg="red")
        elif ok:
            self._log("已发送回零命令 — 抬升→平移→下降至原点")
            self.validation_label.config(text="⟲ 回零中...", fg="green")
        else:
            self._log(f"错误: {resp}")

    def _toggle_obstacle(self):
        """切换障碍物避障开关 (发送 SetObstacle 到 Dashboard)。"""
        self._obs_active = not self._obs_active
        active_int = 1 if self._obs_active else 0
        pos = [0.25, -0.05, 0.28]  # 默认世界坐标位置
        cmd = f"SetObstacle(active={active_int},pos={{{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}}})"
        ok, resp = self._send_dashboard_cmd(cmd)
        if ok:
            state = "已激活" if self._obs_active else "已关闭"
            self.obs_btn.config(text="○ 关闭障碍物" if self._obs_active else "● 激活障碍物")
            self.obs_status_label.config(text=state,
                                         foreground="red" if self._obs_active else "gray")
            self.obs_pos_label.config(text=f"位置: ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})m")
            self._log(f"障碍物避障: {state} (世界坐标 {pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}m)")
        else:
            self._obs_active = not self._obs_active  # 恢复
            self._log(f"障碍物切换失败: {resp}")
            self._log(f"错误: {resp}")

    # ── 外部接口 (大模型坐标输入) ──

    def _start_ext_server(self):
        """启动外部接口 TCP 服务 (独立线程), 接收大模型发送的坐标。"""
        if self.ext_port <= 0:
            return
        self._ext_server_thread = threading.Thread(
            target=self._ext_server, daemon=True)
        self._ext_server_thread.start()
        self._log(f"外部接口: localhost:{self.ext_port} (接收大模型坐标)")

    def _ext_server(self):
        """外部 TCP 服务: 接收 JSON → 填入坐标 → 可选自动发送。"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._ext_server_sock = server
        try:
            server.bind(("localhost", self.ext_port))
            server.listen(1)
            server.settimeout(1.0)
        except OSError as e:
            self._log(f"外部接口启动失败 (端口{self.ext_port}被占用?): {e}")
            self._ext_server_sock = None
            return

        while True:
            conn = None
            try:
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue

                data = conn.recv(4096)
                if not data:
                    conn.close()
                    continue

                req = json.loads(data.decode("utf-8"))
                cmd = req.get("command", "move")  # "move" 或 "obstacle"

                if cmd == "obstacle":
                    # 转发障碍物控制命令到 Dashboard
                    active = req.get("active", None)
                    pos = req.get("pos", None)
                    dash_cmd = "SetObstacle("
                    parts = []
                    if active is not None:
                        parts.append(f"active={int(active)}")
                    if pos is not None and len(pos) >= 3:
                        parts.append(f"pos={{{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}}}")
                    dash_cmd += ",".join(parts) + ")"
                    resp = self._send_cmd(dash_cmd)
                    conn.sendall(json.dumps({"status": "ok", "resp": resp}).encode() + b"\n")
                    conn.close()
                    continue

                # 默认: 坐标移动
                x_mm = float(req["x"])
                y_mm = float(req["y"])
                z_mm = float(req["z"])
                auto_send = req.get("send", self._ext_auto_send)

                # 切回主线程操作 GUI
                self.window.after(0, lambda x=x_mm, y=y_mm, z=z_mm, s=auto_send:
                                  self._apply_external_target(x, y, z, s))

                conn.sendall(json.dumps({"status": "ok"}).encode() + b"\n")
                conn.close()
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if conn:
                    conn.sendall(json.dumps(
                        {"status": "error", "msg": str(e)}).encode() + b"\n")
                    conn.close()
            except OSError:
                if conn:
                    conn.close()

    def _apply_external_target(self, x_mm, y_mm, z_mm, auto_send):
        """将外部坐标填入输入框, 可选自动发送。"""
        self.x_var.set(str(int(x_mm)))
        self.y_var.set(str(int(y_mm)))
        self.z_var.set(str(int(z_mm)))
        self._log(f"外部输入: X={x_mm} Y={y_mm} Z={z_mm}mm")
        if auto_send:
            self._send_target()

    # ── 轮询 ──

    def _poll_position(self):
        """定时查询 TCP 位置 (每 200ms), 断线时自动重连。"""
        if self.sock is not None:
            resp = self._send_cmd("GetPose()")
            pose = self._parse_pose_resp(resp)
            if pose and len(pose) >= 3:
                self.x_pos_label.config(text=f"{pose[0]:.1f}")
                self.y_pos_label.config(text=f"{pose[1]:.1f}")
                self.z_pos_label.config(text=f"{pose[2] - self._z_initial_mm:.1f}")
                self.error_label.config(text=f"TCP(基座): {pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f}mm")
                # 检查 RobotMode
                mode_resp = self._send_cmd("RobotMode()")
                if mode_resp and mode_resp.strip() == "7":
                    self.reached_label.config(text="● 运动中", foreground="orange")
                else:
                    self.reached_label.config(text="✓ 就绪", foreground="green")
            elif resp is not None:
                pass  # 格式不对, 忽略
        else:
            # 断线自动重连 (每 2 秒尝试一次)
            if self._reconnect_cooldown <= 0:
                self._reconnect_cooldown = 10  # 10 * 200ms = 2s
                self.conn_label.config(text="⟳ 重连中...", foreground="orange")
                self._log("自动重连中...")
                self._connect()
            else:
                self._reconnect_cooldown -= 1

        self.window.after(200, self._poll_position)

    def _log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"  {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def run(self):
        self.window.mainloop()


def main():
    parser = argparse.ArgumentParser(description="MG400 控制面板")
    parser.add_argument("--host", default="localhost", help="仿真服务器地址")
    parser.add_argument("--port", type=int, default=29999, help="Dashboard 端口 (29999)")
    parser.add_argument("--ext-port", type=int, default=9878,
                        help="外部接口端口 (大模型坐标输入, 0=禁用)")
    args = parser.parse_args()
    app = MG400ControlPanel(host=args.host, port=args.port, ext_port=args.ext_port)
    app.run()


if __name__ == "__main__":
    main()
