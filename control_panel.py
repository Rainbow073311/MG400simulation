"""
MG400 控制面板 — 独立运行的图形化交互脚本
与 MuJoCo 仿真中的 TCP 服务器 (simulate_slider.py) 通信,
发送目标坐标, 驱动机械臂通过 IK 到达指定位置。

用法:
    python control_panel.py
    python control_panel.py --host localhost --port 9876
"""
import argparse
import json
import socket
import sys
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np


class MG400ControlPanel:
    """控制面板: TCP 客户端 + tkinter GUI"""

    def __init__(self, host="localhost", port=9876, ext_port=9878):
        self.host = host
        self.port = port
        self.ext_port = ext_port
        self.sock = None
        self._reconnect_cooldown = 0
        self._speed_after_id = None
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
        target_frame = ttk.LabelFrame(self.window, text="目标坐标 (X/Y世界, Z相对初始位置, 毫米)", padding=10)
        target_frame.pack(fill="x", padx=12, pady=8)

        # 工作范围 — 由仿真服务器动态提供 (相对 Z 坐标系)
        self.Z_RANGE = (-200, 230)
        self.Z_OPTIMAL = 10
        self.MAX_RADIUS = 440
        self.R_Z_SLOPE = 0.35
        # 精细边界数据 (从 getpos 获取)
        self.ws_z_rel = None    # Z 数组 (相对, mm)
        self.ws_r_min = None    # R_min 数组 (mm)
        self.ws_r_max = None    # R_max 数组 (mm)

        def make_input(parent, label, initial):
            row = ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=label, width=4).pack(side="left")
            var = tk.StringVar(value=initial)
            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side="left")
            hint = {"X:": "mm 世界坐标", "Y:": "mm 世界坐标", "Z:": "mm 相对初始位置"}.get(label, "")
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

    # ── 网络通信 ──

    def _send_cmd(self, cmd_dict):
        """发送 JSON 命令, 返回响应字典。连接失败返回 None。"""
        if self.sock is None:
            return None
        try:
            msg = json.dumps(cmd_dict) + "\n"
            self.sock.sendall(msg.encode("utf-8"))
            resp = self.sock.recv(4096)
            if not resp:
                self._disconnect()
                return None
            return json.loads(resp.decode("utf-8"))
        except (ConnectionResetError, BrokenPipeError, OSError, socket.timeout,
                json.JSONDecodeError) as e:
            self._disconnect()
            return None

    def _connect(self):
        """连接仿真服务器"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(2.0)
            self.sock.connect((self.host, self.port))
            self.sock.settimeout(0.5)
            self.conn_label.config(text="● 已连接", foreground="green")
            self.move_btn.config(state="normal")
            self.stop_btn.config(state="normal")
            self._log(f"已连接到 {self.host}:{self.port}")
            self.validation_label.config(text="✓ 已连接, 可输入坐标", fg="green")
            # 同步当前速度设置
            self._send_cmd({"cmd": "speed", "value": self.speed_var.get()})
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
        """使用精细边界数据验证坐标是否在工作空间内。返回 (ok, msg)。"""
        r = (x_mm**2 + y_mm**2)**0.5

        # J1 方位角校验: 基座固定在世界坐标系 45°, J1 范围 ±160°
        # 可达世界方位角 = [45°-160°, 45°+160°] = [-115°, 205°]
        theta_world = np.degrees(np.arctan2(y_mm, x_mm))
        j1_needed = (theta_world - 45 + 180) % 360 - 180  # 归一化到 [-180, 180]
        if abs(j1_needed) > 160:
            return False, (f"方位角 θ={theta_world:.0f}° 超出J1范围 "
                           f"(基座45°, J1仅±160°): J1需转{j1_needed:.0f}°")

        # 有精细边界数据 → 精确验证
        if self.ws_z_rel is not None and len(self.ws_z_rel) > 2:
            z_arr = np.array(self.ws_z_rel, dtype=float)
            r_min_arr = np.array(self.ws_r_min, dtype=float)
            r_max_arr = np.array(self.ws_r_max, dtype=float)

            if z_mm < z_arr[0] or z_mm > z_arr[-1]:
                return False, f"Z={z_mm:.0f}mm 超出范围 ({z_arr[0]:.0f} ~ {z_arr[-1]:.0f})"

            r_min_z = np.interp(z_mm, z_arr, r_min_arr)
            r_max_z = np.interp(z_mm, z_arr, r_max_arr)

            if r < r_min_z:
                return False, f"r={r:.0f}mm < R_min={r_min_z:.0f}mm (Z={z_mm:.0f}mm 处太靠近基座)"
            if r > r_max_z:
                return False, f"r={r:.0f}mm > R_max={r_max_z:.0f}mm (Z={z_mm:.0f}mm 处超出最大工作半径)"
            return True, ""

        # 无精细数据 → 用简化公式
        if r > self.MAX_RADIUS:
            return False, f"r={r:.0f} > {self.MAX_RADIUS} mm"
        r_max_z = max(50, self.MAX_RADIUS - self.R_Z_SLOPE * abs(z_mm - self.Z_OPTIMAL))
        if r > r_max_z:
            return False, f"Z={z_mm:.0f}mm 处最大可到 r={r_max_z:.0f}mm, 当前 r={r:.0f}mm"
        if not (self.Z_RANGE[0] <= z_mm <= self.Z_RANGE[1]):
            return False, f"Z={z_mm:.0f} 超出范围 ({self.Z_RANGE[0]} ~ {self.Z_RANGE[1]})"
        return True, ""

    def _clear_validation(self):
        """输入框内容变化时清除验证提示"""
        self.validation_label.config(text="", fg="red")

    def _send_target(self):
        """发送目标坐标 (先验证, 再发送)"""
        if self.sock is None:
            self._log("错误: 未连接到仿真")
            self.validation_label.config(text="⚠ 未连接到仿真, 请先点击「连接」", fg="red")
            return
        try:
            x = float(self.x_var.get()) / 1000.0
            y = float(self.y_var.get()) / 1000.0
            z = float(self.z_var.get()) / 1000.0
        except ValueError:
            self._log("错误: 坐标值必须是数字")
            self.validation_label.config(text="⚠ 坐标值必须是数字", fg="red")
            return

        x_mm = x * 1000
        y_mm = y * 1000
        z_mm = z * 1000

        # 工作空间验证
        ok, msg = self._check_workspace(x_mm, y_mm, z_mm)
        if not ok:
            self._log(f"✗ 超出工作范围: {msg}")
            self.validation_label.config(text=f"⚠ 超出工作范围 — {msg}", fg="red")
            return

        self._log(f"✓ 坐标在工作范围内, 发送目标...")
        self.validation_label.config(text="✓ 坐标有效, 正在发送...", fg="green")
        resp = self._send_cmd({"cmd": "move", "x": x, "y": y, "z": z, "speed": float(self.speed_var.get())})
        if resp is None:
            self._log("错误: 发送失败, 请重新连接")
            self.validation_label.config(text="⚠ 发送失败, 请重新连接", fg="red")
            return
        if resp.get("status") == "ok":
            self._log(f"目标已发送: X={x:.4f}  Y={y:.4f}  Z={z:.4f} (相对)")
            self.validation_label.config(text=f"✓ 目标已发送 X={x_mm:.0f} Y={y_mm:.0f} Z={z_mm:.0f}mm", fg="green")
        else:
            err_msg = resp.get('msg', '未知错误')
            self._log(f"错误: {err_msg}")
            self.validation_label.config(text=f"⚠ 服务器拒绝 — {err_msg}", fg="red")

    def _on_speed_change(self, val):
        """速度滑块变化时发送到仿真 (防抖: 停止拖动150ms后才发送)"""
        pct = int(float(val) * 100)
        self.speed_label.config(text=f"{pct}%")
        if self._speed_after_id is not None:
            self.window.after_cancel(self._speed_after_id)
        self._speed_after_id = self.window.after(
            150, lambda: self._send_cmd({"cmd": "speed", "value": self.speed_var.get()}))

    def _stop(self):
        """停止当前运动"""
        resp = self._send_cmd({"cmd": "stop"})
        if resp and resp.get("status") == "ok":
            self._log("已发送停止命令")

    def _go_home(self):
        """回零 — 返回安全原点姿态 (J1=J2=J3=J4=0)"""
        if self.sock is None:
            self._log("错误: 未连接到仿真")
            self.validation_label.config(text="⚠ 未连接到仿真", fg="red")
            return
        resp = self._send_cmd({"cmd": "home", "speed": float(self.speed_var.get())})
        if resp is None:
            self._log("错误: 发送失败, 请重新连接")
            self.validation_label.config(text="⚠ 发送失败", fg="red")
        elif resp.get("status") == "ok":
            self._log("已发送回零命令 — 抬升→平移→下降至原点")
            self.validation_label.config(text="⟲ 回零中...", fg="green")
        else:
            self._log(f"错误: {resp.get('msg', '未知')}")

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
            resp = self._send_cmd({"cmd": "getpos"})
            if resp is not None and "tcp" in resp:
                tcp = resp["tcp"]
                err = resp["error"]
                reached = resp["reached"]
                self.x_pos_label.config(text=f"{tcp[0]*1000:.1f}")
                self.y_pos_label.config(text=f"{tcp[1]*1000:.1f}")
                self.z_pos_label.config(text=f"{tcp[2]*1000:.1f}")
                self.error_label.config(text=f"误差: {err*1000:.0f} mm")
                if reached:
                    self.reached_label.config(text="✓ 已到达", foreground="green")
                else:
                    self.reached_label.config(text="● 运动中", foreground="orange")
                if "z_range_rel" in resp:
                    self.Z_RANGE = (resp["z_range_rel"][0], resp["z_range_rel"][1])
                if "z_optimal_rel" in resp:
                    self.Z_OPTIMAL = resp["z_optimal_rel"]
                if "ws_z_rel" in resp:
                    self.ws_z_rel = resp["ws_z_rel"]
                    self.ws_r_min = resp["ws_r_min"]
                    self.ws_r_max = resp["ws_r_max"]
            else:
                pass  # _send_cmd 内部已调用 _disconnect
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
    parser.add_argument("--port", type=int, default=9876, help="仿真服务器端口")
    parser.add_argument("--ext-port", type=int, default=9878,
                        help="外部接口端口 (大模型坐标输入, 0=禁用)")
    args = parser.parse_args()
    app = MG400ControlPanel(host=args.host, port=args.port, ext_port=args.ext_port)
    app.run()


if __name__ == "__main__":
    main()
