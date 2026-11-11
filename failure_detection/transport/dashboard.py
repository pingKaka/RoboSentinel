"""Local OpenCV dashboard; images and overlays always share a timestamp."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DetectionDashboard:
    def __init__(self, font_path=None):
        candidates = [font_path] if font_path else [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            "C:/Windows/Fonts/msyh.ttc",
        ]
        path = next((str(p) for p in candidates if p and Path(p).is_file()), None)
        if path is None:
            raise RuntimeError("Install fonts-noto-cjk or set sentinel.dashboard_font to a Chinese font file")
        self.font = ImageFont.truetype(path, 20)
        self.title_font = ImageFont.truetype(path, 27)
        self.scroll = 0
        self.last_fault = None
        self.line_count = 0
        self._body = None
        self._lines = []
        self.info_scroll = 0
        self.info_lines = []
        self._info = None

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if x >= 1200:
                self.info_scroll = max(0, min(max(0, len(self.info_lines) - 29), self.info_scroll + (-3 if flags > 0 else 3)))
                return
            self.scroll = max(0, min(max(0, self.line_count - 7), self.scroll + (-3 if flags > 0 else 3)))

    def render(self, frames, client=None, active=False, now=0, notice="", keys=("space", "f", "esc"), task=None):
        canvas = Image.new("RGB", (1600, 940), "#f2f4f7")
        draw = ImageDraw.Draw(canvas)
        fault = client.fault_result if client else None
        visual = client.latest_visual if client else None
        status, color = "未开始检测", "#52616b"
        if active:
            status, color = "初始化：理解场景与任务", "#976500"
            if client and client.status == "streaming":
                status, color = "检测中", "#168052"
            if client and client._manual_busy:
                status, color = "发生故障 · 人工报告提交中", "#b63636"
            if fault:
                status, color = "发生故障", "#b63636"
            if client and client.error:
                status, color = "检测连接异常", "#b63636"
        draw.rectangle((0, 0, 1600, 65), fill=color)
        draw.text((20, 13), "RoboSentinel  /  " + status, font=self.title_font, fill="white")
        for index, camera in enumerate(("external", "wrist")):
            x = 20 + index * 600
            draw.text((x, 76), ("外部相机" if index == 0 else "腕部相机") + " · 实时原图", font=self.font, fill="#202b33")
            self._image(canvas, frames[camera], (x, 108, 560, 190))
            draw.text((x, 309), "DINO + SAM · 已处理画面", font=self.font, fill="#202b33")
            if visual:
                result, images = visual
                image = images[camera].copy()
                for track in (result.get("perception") or {}).get("tracks", []):
                    if track["camera"] != camera or not track["visible"]:
                        continue
                    contours = [np.asarray(c, dtype=np.int32).reshape(-1, 1, 2) for c in track.get("contours", []) if len(c) >= 3]
                    if contours:
                        overlay = image.copy()
                        cv2.fillPoly(overlay, contours, (40, 200, 140))
                        image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
                        cv2.polylines(image, contours, True, (40, 230, 140), 2)
                    if track.get("box_xyxy"):
                        x1, y1, x2, y2 = map(int, track["box_xyxy"])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 195, 40), 2)
                        cv2.putText(image, track["target_id"], (max(0, x1), max(15, y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 195, 40), 1)
                self._image(canvas, image, (x, 345, 560, 250))
            else:
                draw.rectangle((x, 345, x + 560, 595), fill="#dce2e8")
                draw.text((x + 18, 450), "等待检测结果", font=self.font, fill="#52616b")
        stamp = visual[0].get("timestamp_sec", 0) if visual else None
        detail = "" if stamp is None else f"处理帧 {stamp:.2f}s  |  距当前 {max(0, now - stamp):.1f}s"
        draw.text((20, 608), detail, font=self.font, fill="#52616b")
        draw.line((20, 642, 1180, 642), fill="#c3cbd2")
        if fault != self.last_fault:
            self.scroll = 0
            self.last_fault = fault
        findings = (fault or {}).get("rules", {}).get("findings", [])
        source = "人工报告" if (fault or {}).get("manual_failure") else "自动规则"
        body = "暂无故障检测结果。"
        if findings:
            sections = []
            for finding in findings:
                confidence = finding.get("confidence")
                lines = [f'规则: {finding.get("rule_id", "--")}',
                         f'故障类型: {finding.get("failure_type") or "未分类"}',
                         f'状态: {finding.get("status", "--")}',
                         f'来源: {source}']
                if confidence is not None:
                    lines.append(f'置信度: {confidence}')
                lines.append("证据: " + json.dumps(finding.get("evidence", {}), ensure_ascii=False, sort_keys=True))
                sections.append("\n".join(lines))
            body = "\n\n".join(sections)
        elif client and client._manual_busy:
            body = "人工故障报告提交中。"
        if client and client.error:
            body = "连接错误：" + client.error + "\n" + body
        if body != self._body:
            self._body, self._lines = body, self._wrap(body, 1140)
        lines = self._lines
        self.line_count = len(lines)
        self.scroll = min(self.scroll, max(0, len(lines) - 7))
        draw.text((20, 650), f"故障检测结果   {self.scroll + 1}-{min(len(lines), self.scroll + 7)}/{len(lines)}", font=self.font, fill="#202b33")
        for index, line in enumerate(lines[self.scroll:self.scroll + 7]):
            draw.text((20, 682 + index * 27), line, font=self.font, fill="#202b33")
        draw.text((20, 875), notice, font=self.font, fill="#976500")
        start, failure, quit_key = keys
        draw.text((20, 910), f"{start}: 开始/停止  |  {failure}: 报告故障  |  {quit_key}: 退出", font=self.font, fill="#52616b")
        self._information(draw, client, task or {}, active, fault)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def _information(self, draw, client, task, active, fault):
        initial = getattr(client, "initial_result", None) or {}
        result = getattr(client, "latest_result", None) or initial
        task_id = initial.get("task_id") or getattr(client, "task_id", None) or task.get("task_id", "--")
        description = initial.get("description") or getattr(client, "description", None) or task.get("description", "未配置")
        targets = (initial.get("tracking_plan") or {}).get("targets", [])
        tracks = (result.get("perception") or {}).get("tracks", [])
        roles = {"manipulated_object": "操作对象", "destination": "目的地", "gripper": "夹爪"}
        lines = ["任务 ID: " + str(task_id), description, "", f"跟踪目标 ({len(targets)})"]
        if not targets:
            lines.append("等待初始场景理解" if active else "开始后由 VLM 自动生成")
        for target in targets:
            lines.extend([f'{target["target_id"]} · {roles.get(target["role"], target["role"])}', target["name"]])
            visible = []
            for camera, label in (("external", "外部"), ("wrist", "腕部")):
                matching = [t for t in tracks if t["target_id"] == target["target_id"] and t["camera"] == camera]
                visible.append(label + ": " + ("未知" if not matching else "可见" if any(t["visible"] for t in matching) else "不可见"))
            lines.append(" / ".join(visible))
        lines.extend(["", "运行信息", "连接: " + str(getattr(client, "status", "未连接")),
                      "服务: " + str(getattr(client, "server", None) or task.get("server", "--")),
                      "会话: " + str(initial.get("task_session_id", "--")),
                      "录像: " + ("录制中" if active else "未录制"),
                      f'已返回: {getattr(client, "responses", 0)}  丢弃旧观测: {getattr(client, "dropped", 0)}',
                      f'模型处理: {result.get("server_processing_ms", "--")} ms',
                      f'请求往返: {result.get("client_roundtrip_ms", "--")} ms',
                      "", "当前规则"])
        findings = (result.get("rules") or {}).get("findings", [])
        states = {"clear": "正常", "unknown": "未知", "confirmed": "确认异常"}
        lines.extend([f'{f["rule_id"]}: {states.get(f["status"], f["status"])}' for f in findings] or ["尚无规则结果"])
        robot = (result.get("features") or {}).get("robot", {})
        lines.extend(["", "机器人特征",
                      "TCP: " + str(robot.get("tcp_position_m") or "未知"),
                      "夹爪位置: " + str(robot.get("gripper_position_normalized")
                                       if robot.get("gripper_position_normalized") is not None else "未知")])
        if fault:
            lines.extend(["", "最近故障", "来源: " + ("人工报告" if fault.get("manual_failure") else "自动规则"),
                          f'观测时间: {fault.get("timestamp_sec", "--")} s'])
        lines.extend(["", "本地结果目录", str(getattr(client, "output_dir", "开始后创建"))])
        info = "\n".join(lines)
        if info != self._info:
            self._info, self.info_lines = info, self._wrap(info, 350)
        self.info_scroll = min(self.info_scroll, max(0, len(self.info_lines) - 29))
        draw.line((1200, 65, 1200, 940), fill="#c3cbd2")
        draw.text((1220, 78), "任务与检测详情", font=self.title_font, fill="#202b33")
        for index, line in enumerate(self.info_lines[self.info_scroll:self.info_scroll + 29]):
            draw.text((1220, 122 + index * 27), line, font=self.font, fill="#202b33")
        draw.text((1220, 914), f'{self.info_scroll + 1}-{min(len(self.info_lines), self.info_scroll + 29)}/{len(self.info_lines)}', font=self.font, fill="#52616b")

    def _wrap(self, text, width):
        lines = []
        for paragraph in text.splitlines():
            line = ""
            for char in paragraph:
                if self.font.getlength(line + char) > width:
                    lines.append(line)
                    line = ""
                line += char
            lines.append(line)
        return lines or [""]

    @staticmethod
    def _image(canvas, array, rect):
        x, y, width, height = rect
        image = Image.fromarray(array)
        image.thumbnail((width, height))
        ImageDraw.Draw(canvas).rectangle((x, y, x + width, y + height), fill="#171c20")
        canvas.paste(image, (x + (width - image.width) // 2, y + (height - image.height) // 2))
