import os
import sys
import csv
import traceback
from glob import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import config as cfg
from test_tools.common import detect_all, grab_all_frames
from test_tools.ct.operations import find_longest, multiple_tracking
from test_tools.faster_crop_align_xray import FasterCropAlignXRay
from test_tools.supply_writer import SupplyWriter
from test_tools.utils import get_crop_box
from utils.plugin_loader import PluginLoader


# ==============================
# 用户可配置区
# ==============================
max_frame = 400
# 可填目录（批量处理）或单个文件
input_path = "examples"
out_dir = "prediction"
cfg_path = "i3d_ori.yaml"
ckpt_path = "checkpoints/model.pth"
optimal_threshold = 0.04

# 支持的视频扩展名
VIDEO_EXTS = {".mp4", ".MP4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_output_path(video_path: str, out_root: str) -> str:
    """将输入视频名改为 .avi，输出到 out_root 下。"""
    base = os.path.splitext(os.path.basename(video_path))[0] + ".avi"
    return os.path.join(out_root, base)


def sanitize_cache_name(video_path: str, max_frame: int) -> str:
    """缓存文件名（保持与原脚本一致，以便复用缓存）"""
    return f"{video_path}_{max_frame}.pth"


def process_one_video(
    video_path: str,
    classifier,
    crop_align_func,
    device: torch.device,
    imsize: int,
):
    """
    处理单个视频：
      - 生成标注视频到 out_dir
      - 返回 (avg_prob, per_frame_list)
        per_frame_list: List[(frame_idx:int, pred_prob:float or None)]
    """
    # mean/std 到正确 device
    mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255], device=device).view(1, 3, 1, 1, 1)

    ensure_dir(out_dir)
    out_file = build_output_path(video_path, out_dir)
    cache_file = sanitize_cache_name(video_path, max_frame)

    # 检测/读取帧
    if os.path.exists(cache_file):
        detect_res, all_lm68 = torch.load(cache_file, map_location="cpu")
        frames = grab_all_frames(video_path, max_size=max_frame, cvt=True)
        print(f"[CACHE] detection result loaded from {cache_file}")
    else:
        print("[DETECT] start:", video_path)
        detect_res, all_lm68, frames = detect_all(video_path, return_frames=True, max_size=max_frame)
        torch.save((detect_res, all_lm68), cache_file)
        print("[DETECT] finished")

    if len(frames) == 0:
        raise RuntimeError("No frames grabbed. Check the video file.")

    print("[INFO] number of frames:", len(frames))
    shape = frames[0].shape[:2]

    # 组装检测结果（加入 68 点）
    all_detect_res = []
    assert len(all_lm68) == len(detect_res)
    for faces, faces_lm68 in zip(detect_res, all_lm68):
        new_faces = []
        for (box, lm5, score), face_lm68 in zip(faces, faces_lm68):
            new_face = (box, lm5, face_lm68, score)
            new_faces.append(new_face)
        all_detect_res.append(new_faces)
    detect_res = all_detect_res

    # 跟踪
    print("[TRACK] split into super clips")
    tracks = multiple_tracking(detect_res)
    tuples = [(0, len(detect_res))] * len(tracks)
    print("[TRACK] full_tracks", len(tracks))
    if len(tracks) == 0:
        tuples, tracks = find_longest(detect_res)

    data_storage = {}
    frame_boxes = {}
    super_clips = []

    for track_i, ((start, end), track) in enumerate(zip(tuples, tracks)):
        print(f"[TRACK] segment {track_i}: {start} -> {end}")
        assert len(detect_res[start:end]) == len(track)

        super_clips.append(len(track))

        for face, frame_idx, j in zip(track, range(start, end), range(len(track))):
            box, lm5, lm68 = face[:3]
            big_box = get_crop_box(shape, box, scale=0.5)

            top_left = big_box[:2][None, :]
            new_lm5 = lm5 - top_left
            new_lm68 = lm68 - top_left
            new_box = (box.reshape(2, 2) - top_left).reshape(-1)
            info = (new_box, new_lm5, new_lm68, big_box)

            x1, y1, x2, y2 = big_box
            cropped = frames[frame_idx][y1:y2, x1:x2]
            base_key = f"{track_i}_{j}_"
            data_storage[f"{base_key}img"] = cropped
            data_storage[f"{base_key}ldm"] = info
            data_storage[f"{base_key}idx"] = frame_idx
            frame_boxes[frame_idx] = np.rint(box).astype(int)

    print("[SAMPLE] sampling clips from super clips", super_clips)

    clips_for_video = []
    clip_size = cfg.clip_size
    pad_length = clip_size - 1

    for super_clip_idx, super_clip_size in enumerate(super_clips):
        inner_index = list(range(super_clip_size))
        if super_clip_size < clip_size:  # padding
            post_module = inner_index[1:-1][::-1] + inner_index
            l_post = len(post_module)
            post_module = post_module * (pad_length // l_post + 1)
            post_module = post_module[:pad_length]
            assert len(post_module) == pad_length

            pre_module = inner_index + inner_index[1:-1][::-1]
            l_pre = len(pre_module)
            pre_module = pre_module * (pad_length // l_pre + 1)
            pre_module = pre_module[-pad_length:]
            assert len(pre_module) == pad_length

            inner_index = pre_module + inner_index + post_module

        super_clip_size = len(inner_index)
        frame_range = [
            inner_index[i: i + clip_size] for i in range(super_clip_size) if i + clip_size <= super_clip_size
        ]
        for indices in frame_range:
            clip = [(super_clip_idx, t) for t in indices]
            clips_for_video.append(clip)

    preds = []
    frame_res = {}

    for clip in tqdm(clips_for_video, desc=f"testing [{os.path.basename(video_path)}]"):
        images = [data_storage[f"{i}_{j}_img"] for i, j in clip]
        landmarks = [data_storage[f"{i}_{j}_ldm"] for i, j in clip]
        frame_ids = [data_storage[f"{i}_{j}_idx"] for i, j in clip]

        _, images_align = crop_align_func(landmarks, images)

        # 可视化临时拼接（不参与推理）
        for i in range(clip_size):
            img1 = cv2.resize(images[i], (imsize, imsize))
            _ = np.concatenate((img1, images_align[i]), axis=1)

        images_t = torch.as_tensor(images_align, dtype=torch.float32, device=device).permute(3, 0, 1, 2)
        images_t = images_t.unsqueeze(0).sub(mean).div(std)

        with torch.no_grad():
            output = classifier(images_t)
        pred = float(torch.sigmoid(output["final_output"]))
        for f_id in frame_ids:
            if f_id not in frame_res:
                frame_res[f_id] = []
            frame_res[f_id].append(pred)
        preds.append(pred)

    avg_prob = float(np.mean(preds)) if len(preds) > 0 else 0.0
    print(f"[RESULT] avg prob = {avg_prob:.6f}")

    # 汇总每帧（平均概率），同时为后续写 CSV 准备行
    boxes = []
    scores = []
    per_frame_list = []
    for frame_idx in range(len(frames)):
        if frame_idx in frame_res:
            pred_prob = float(np.mean(frame_res[frame_idx]))
            rect = frame_boxes[frame_idx]
        else:
            pred_prob = None
            rect = None
        scores.append(pred_prob)
        boxes.append(rect)
        per_frame_list.append((frame_idx, pred_prob))

    # 写出带标注视频
    SupplyWriter(video_path, out_file, optimal_threshold).run(frames, scores, boxes)
    return avg_prob, per_frame_list


def main():
    # 初始化配置
    cfg.init_with_yaml()
    cfg.update_with_yaml(cfg_path)
    cfg.freeze()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] Using {device}")

    # 分类器
    classifier = PluginLoader.get_classifier(cfg.classifier_type)()
    classifier.to(device)
    classifier.eval()
    classifier.load(ckpt_path)

    # 对齐器
    crop_align_func = FasterCropAlignXRay(cfg.imsize)

    # 收集要处理的视频列表
    if os.path.isdir(input_path):
        vids = []
        for ext in VIDEO_EXTS:
            vids.extend(glob(os.path.join(input_path, f"**/*{ext}"), recursive=True))
        vids = sorted(vids)
    else:
        if not is_video_file(input_path):
            print(f"[WARN] Not a supported video: {input_path}")
            return
        vids = [input_path]

    if not vids:
        print(f"[WARN] No videos found in: {input_path}")
        return

    ensure_dir(out_dir)
    # 两个 CSV 路径
    csv_summary = os.path.join(out_dir, "results.csv")
    csv_frames = os.path.join(out_dir, "frame_results.csv")

    # 先写入表头
    with open(csv_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "avg_prob"])

    with open(csv_frames, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_name", "video_path", "frame_index", "pred_prob"])

    # 逐个处理并写入 CSV
    for vp in vids:
        abs_path = os.path.abspath(vp)
        filename = os.path.basename(vp)
        try:
            avg_prob, per_frame_list = process_one_video(
                video_path=vp,
                classifier=classifier,
                crop_align_func=crop_align_func,
                device=device,
                imsize=cfg.imsize,
            )
            # 写 summary CSV
            with open(csv_summary, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([filename, abs_path, avg_prob])

            # 写逐帧 CSV（逐帧一行；pred_prob 可能为 None）
            with open(csv_frames, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for frame_idx, prob in per_frame_list:
                    writer.writerow([filename, abs_path, frame_idx, "" if prob is None else prob])

        except Exception as e:
            print(f"[ERROR] failed on {vp}: {e}")
            traceback.print_exc()
            # 失败也在 summary 里记一笔
            with open(csv_summary, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([filename, abs_path, "ERROR"])

    print(f"[DONE] Processed {len(vids)} video(s).")
    print(f"  - Summary CSV: {csv_summary}")
    print(f"  - Frame CSV:   {csv_frames}")


if __name__ == "__main__":
    main()
