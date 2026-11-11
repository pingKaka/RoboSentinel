#!/usr/bin/env python3
"""Render an inspectable HTML/contact-sheet report from collected video features."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


COLORS = {
    "bottle": (70, 190, 80),
    "cup": (55, 145, 235),
    "gripper": (210, 120, 55),
}
CAMERAS = ("external", "wrist")


def _text(image, value, origin, color=(255, 255, 255), scale=0.48, thickness=1):
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2,
                cv2.LINE_AA)
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv2.LINE_AA)


def _plain_text(image, value, origin, color, scale=0.45):
    cv2.putText(image, value, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _read_frame(video: Path, frame_index: int):
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Cannot read frame {frame_index} from {video}")
        return frame
    finally:
        capture.release()


def _camera_features(snapshot: dict, camera: str):
    for entity in snapshot["entities"]:
        feature = next(item for item in entity["cameras"] if item["camera"] == camera)
        yield entity, feature


def _draw_camera(frame, snapshot: dict, camera: str):
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    unknown = []
    not_visible = []
    labels = []
    for entity, feature in _camera_features(snapshot, camera):
        target_id = entity["target_id"]
        visibility = feature["visibility"]
        if visibility == "unknown":
            unknown.append(target_id)
            labels.append((f"{target_id} unknown", (155, 155, 155)))
            continue
        if visibility == "not_visible":
            not_visible.append(target_id)
            labels.append((f"{target_id} not-visible", (155, 155, 155)))
            continue
        center = feature["mask_centroid_xy"]
        if center is None:
            unknown.append(target_id)
            labels.append((f"{target_id} unknown", (155, 155, 155)))
            continue
        x = int(center[0] * width)
        y = int(center[1] * height)
        color = COLORS.get(target_id, (170, 170, 170))
        cv2.drawMarker(canvas, (x, y), color, cv2.MARKER_CROSS, 18, 2)
        quality = feature["tracking_quality"]
        labels.append((target_id if quality is None else f"{target_id} {quality:.2f}", color))

    header = np.full((48, width, 3), 35, dtype=np.uint8)
    status = []
    if unknown:
        status.append("unknown: " + ", ".join(unknown))
    if not_visible:
        status.append("not visible: " + ", ".join(not_visible))
    _text(header, camera + (" | " + " | ".join(status) if status else " | all observations known"),
          (7, 17), color=(220, 220, 220), scale=0.4)
    x = 7
    for label, color in labels:
        _text(header, label, (x, 39), color=color, scale=0.4)
        text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
        x += text_width + 16
    return cv2.vconcat((header, canvas))


def _render_row(video_dir: Path, record: dict):
    label = record["video"]
    frame_index = int(record["frame_index"])
    snapshot = record["features"]
    panels = []
    for camera in CAMERAS:
        path = video_dir / f"{'exterior' if camera == 'external' else 'wrist'}_{label}.mp4"
        panels.append(_draw_camera(_read_frame(path, frame_index), snapshot, camera))
    target_height = min(panel.shape[0] for panel in panels)
    resized = [cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height))
               for panel in panels]
    body = cv2.hconcat(resized)
    header = np.full((50, body.shape[1], 3), 28, dtype=np.uint8)
    slots = sum(len(entity["cameras"]) for entity in snapshot["entities"])
    known = sum(feature["visibility"] != "unknown" for entity in snapshot["entities"]
                for feature in entity["cameras"])
    title = (f"{label}  frame={frame_index}  t={snapshot['timestamp_sec']:.2f}s  "
             f"known={known}/{slots}  relations={len(snapshot['relations'])}")
    _text(header, title, (10, 22), scale=0.52)
    _text(header, "cross=mask centroid  green=bottle  orange=cup  blue=gripper  number=DINO init quality",
          (10, 42), color=(190, 190, 190), scale=0.4)
    return cv2.vconcat((header, body))


def _make_sheet(rows, output: Path, video_dir: Path):
    rendered = [_render_row(video_dir, record) for record in rows]
    width = max(row.shape[1] for row in rendered)
    padded = []
    for row in rendered:
        if row.shape[1] < width:
            row = cv2.copyMakeBorder(row, 0, 0, 0, width - row.shape[1], cv2.BORDER_CONSTANT,
                                     value=(28, 28, 28))
        padded.append(row)
    sheet = cv2.vconcat(padded)
    if not cv2.imwrite(str(output), sheet):
        raise OSError(f"Cannot write report image: {output}")


def _select_rows(rows, maximum: int):
    if len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    return [rows[index] for index in indices]


def _make_timeline(rows, output: Path):
    label_width, chart_width, row_height, header_height = 190, 1200, 38, 48
    canvas = np.full((6 * row_height + header_height, label_width + chart_width, 3),
                     245, dtype=np.uint8)
    canvas[:, :label_width] = (42, 42, 42)
    slots = [(camera, target) for camera in CAMERAS for target in ("bottle", "cup", "gripper")]
    status_colors = {
        "visible": (80, 180, 90),
        "not_visible": (55, 155, 230),
        "unknown": (155, 155, 155),
    }
    for row_index, (camera, target_id) in enumerate(slots):
        top = header_height + row_index * row_height
        _plain_text(canvas, f"{camera} / {target_id}", (10, top + 25), (235, 235, 235), 0.47)
        for column, record in enumerate(rows):
            snapshot = record["features"]
            entity = next(item for item in snapshot["entities"] if item["target_id"] == target_id)
            feature = next(item for item in entity["cameras"] if item["camera"] == camera)
            left = label_width + round(column * chart_width / len(rows))
            right = label_width + round((column + 1) * chart_width / len(rows))
            cv2.rectangle(canvas, (left, top), (max(left, right - 1), top + row_height - 2),
                          status_colors[feature["visibility"]], -1)
    _plain_text(canvas, "one column per source frame", (label_width + 5, 29), (50, 50, 50), 0.48)
    legend_x = label_width + 470
    for status in ("visible", "not_visible", "unknown"):
        cv2.rectangle(canvas, (legend_x, 15), (legend_x + 15, 30), status_colors[status], -1)
        _plain_text(canvas, status.replace("_", " "), (legend_x + 22, 29), (50, 50, 50), 0.42)
        legend_x += 135
    if not cv2.imwrite(str(output), canvas):
        raise OSError(f"Cannot write timeline image: {output}")


def _metric(value):
    return f"{100 * value:.1f}%"


def _write_html(output: Path, summary: dict, image_names: dict[str, str], timeline_names: dict[str, str]):
    total = summary["total"]
    cards = []
    for label, metrics in summary["videos"].items():
        cards.append(f"""
        <section>
          <h2>{html.escape(label.replace('_', ' ').title())}</h2>
          <div class="metrics">
            <span>可靠观测 <b>{metrics['visibility_known']}/{metrics['entity_camera_slots']}</b></span>
            <span>覆盖率 <b>{_metric(metrics['visibility_known_rate'])}</b></span>
            <span>空间关系 <b>{metrics['relations']}</b></span>
          </div>
          <h3>逐帧观测状态</h3>
          <a href="{html.escape(timeline_names[label])}"><img src="{html.escape(timeline_names[label])}"
             alt="{html.escape(label)} per-frame feature timeline"></a>
          <h3>均匀抽样标注帧</h3>
          <a href="{html.escape(image_names[label])}"><img src="{html.escape(image_names[label])}"
             alt="{html.escape(label)} feature contact sheet"></a>
        </section>""")
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>标准特征视频测试报告</title>
  <style>
    body {{ margin: 0; font: 15px/1.5 system-ui, sans-serif; color: #202124; background: #f4f5f6; }}
    header {{ padding: 28px max(24px, calc((100% - 1120px)/2)); color: white; background: #202124; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }} h2 {{ margin: 0; font-size: 19px; }}
    header p {{ margin: 0; color: #c5c8ca; }}
    main {{ max-width: 1120px; margin: auto; padding: 24px; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; margin-bottom: 24px;
                border: 1px solid #d8dadd; background: #d8dadd; }}
    .summary div {{ padding: 16px; background: white; }} .summary b {{ display: block; font-size: 23px; }}
    section {{ margin: 0 0 24px; padding: 18px; border: 1px solid #d8dadd; background: white; }}
    .metrics {{ display: flex; gap: 24px; margin: 6px 0 14px; color: #60656a; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #202124; }}
    .note {{ margin-bottom: 24px; padding: 14px 16px; border-left: 4px solid #e09532; background: #fff8ec; }}
    @media (max-width: 700px) {{ .summary {{ grid-template-columns: 1fr 1fr; }} .metrics {{ display: block; }} }}
  </style>
</head>
<body>
<header><h1>标准特征逐帧验证报告</h1><p>双相机逐帧输入 · GroundingDINO 初始化 · SAM 跟踪 · v{summary['schema_version']} 标准化 · 不进行故障判断</p></header>
<main>
  <div class="summary">
    <div><b>{total['snapshots']}</b>特征快照</div>
    <div><b>{_metric(total['vision_valid_rate'])}</b>视觉帧有效率</div>
    <div><b>{total['visibility_known']}/{total['entity_camera_slots']}</b>可靠实体观测</div>
    <div><b>{total['relations']}</b>空间关系</div>
  </div>
  <div class="note"><b>验证范围：</b>本报告只检查每帧标准特征能否生成，不判断视频是否发生故障。
  时间轴每一列对应一个源帧；<code>unknown</code> 表示该帧仍有结构化特征记录，但模型没有形成可靠观测。
  纯视频不含 TCP 与夹爪数据，因此机器人特征有效率为 0。</div>
  {''.join(cards)}
</main>
</body>
</html>
"""
    output.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=Path("result/feature_collection/video_features.jsonl"))
    parser.add_argument("--summary", type=Path,
                        default=Path("result/feature_collection/video_features.summary.json"))
    parser.add_argument("--video-dir", type=Path, default=Path("video"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("result/feature_collection/visual_report"))
    parser.add_argument("--max-images-per-video", type=int, default=8,
                        help="Maximum annotated frames in each contact sheet.")
    args = parser.parse_args()

    records = [json.loads(line) for line in args.features.read_text(encoding="utf-8").splitlines()]
    if not records:
        parser.error(f"No feature records in {args.features}")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["video"]].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_names = {}
    timeline_names = {}
    for label, rows in sorted(grouped.items()):
        name = f"{label}.jpg"
        selected = _select_rows(rows, args.max_images_per_video)
        _make_sheet(selected, args.output_dir / name, args.video_dir)
        image_names[label] = name
        timeline_name = f"{label}_timeline.png"
        _make_timeline(rows, args.output_dir / timeline_name)
        timeline_names[label] = timeline_name
        print(f"rendered {label}: {len(rows)} frame states, {len(selected)} annotated frames")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    _write_html(args.output_dir / "index.html", summary, image_names, timeline_names)
    print(f"report={args.output_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
