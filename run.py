#!/usr/bin/env python3
import sys
import os
import shutil
import glob
import time
import multiprocessing as mp
from pathlib import Path
import psutil
import cv2

from core.processor import process_video, process_img
from core.utils import is_img, detect_fps, set_fps, create_video, add_audio, extract_frames, rreplace
from core.config import get_face
import core.globals

# ------------------ 检查 ffmpeg ------------------
if not shutil.which('ffmpeg'):
    print('ffmpeg is not installed. Read the docs: https://github.com/s0md3v/roop#installation.\n'*3)
    quit()

# ------------------ GPU 设置 ------------------
if core.globals.use_gpu:
    import torch
    if not torch.cuda.is_available():
        print("[WARNING] GPU not available, fallback to CPU.")
        core.globals.use_gpu = False
        core.globals.providers = ['CPUExecutionProvider']

# ------------------ 命令行参数 ------------------
import argparse
parser = argparse.ArgumentParser(description="Roop CLI - 多人脸替换（支持 skip 保留原样）")
parser.add_argument('-f', '--face', dest='source_imgs', nargs='+', help='使用的人脸图片列表，可用 skip/none 保留原样')
parser.add_argument('-t', '--target', dest='target_path', required=True, help='目标视频/图片路径')
parser.add_argument('-o', '--output', dest='output_file', help='输出路径')
parser.add_argument('--keep-fps', action='store_true', help='保持原视频帧率')
parser.add_argument('--keep-frames', action='store_true', help='保留中间帧目录')
parser.add_argument('--cores', dest='cores_count', type=int, help='多进程核心数')
parser.add_argument('--from-right', action='store_true', help='从右往左数人脸')

args = parser.parse_args()

if not args.source_imgs or len(args.source_imgs) == 0:
    print("[WARNING] 请提供人脸图片列表 -f face1.jpg face2.jpg ...")
    quit()

if not args.cores_count:
    args.cores_count = max(1, psutil.cpu_count() - 1)

sep = "/" if os.name != "nt" else "\\"

pool = mp.Pool(args.cores_count)

def start_processing(frame_paths):
    start_time = time.time()
    if core.globals.use_gpu:
        # GPU 单进程处理
        process_video(args.source_imgs, frame_paths)
    else:
        # CPU 多进程处理
        n = len(frame_paths)//(args.cores_count)
        processes = []
        for i in range(0, len(frame_paths), n):
            p = pool.apply_async(process_video, args=(args.source_imgs, frame_paths[i:i+n]))
            processes.append(p)
        for p in processes:
            p.get()
    end_time = time.time()
    print(f"\nProcessing time: {end_time - start_time:.2f} seconds")

def main():
    if not os.path.isfile(args.target_path):
        print("[ERROR] 目标文件不存在:", args.target_path)
        return

    if not args.output_file:
        base = os.path.basename(args.target_path)
        args.output_file = rreplace(args.target_path, sep, sep + "swapped-", 1) if sep in args.target_path else "swapped-" + base

    # 测试人脸有效性
    test_face_img = next((f for f in args.source_imgs if f.lower() not in ["skip","none"]), None)
    if test_face_img and not get_face(cv2.imread(test_face_img)):
        print(f"[WARNING] 测试人脸 {test_face_img} 未检测到，请检查图片。")
        return

    # 图片处理
    if is_img(args.target_path):
        process_img(args.source_imgs, args.target_path, args.output_file)
        print("Swap successful!")
        return

    # 视频处理
    video_name = os.path.splitext(os.path.basename(args.target_path))[0]
    output_dir = os.path.join(os.path.dirname(args.target_path), video_name)
    Path(output_dir).mkdir(exist_ok=True)

    # 检测帧率
    fps = detect_fps(args.target_path)
    tmp_path = args.target_path
    if not args.keep_fps and fps > 30:
        tmp_path = os.path.join(output_dir, video_name + "_30fps.mp4")
        set_fps(args.target_path, tmp_path, 30)
        fps = 30

    shutil.copy(tmp_path, output_dir)
    extract_frames(tmp_path, output_dir)

    args.frame_paths = tuple(sorted(
        glob.glob(os.path.join(output_dir, "*.png")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    ))

    start_processing(args.frame_paths)
    create_video(video_name, fps, output_dir)
    add_audio(output_dir, tmp_path, args.keep_frames, args.output_file)

    print("\nVideo saved as:", args.output_file)
    print("Swap successful!")

if __name__ == "__main__":
    main()
