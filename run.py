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

# 设置多进程启动方式
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

# ------------------ 检查 ffmpeg ------------------
if not shutil.which('ffmpeg'):
    print('ffmpeg is not installed. Read the docs: https://github.com/s0md3v/roop#installation.\n'*3)
    quit()

# ------------------ 环境检测 ------------------
def detect_environment():
    """检测运行环境"""
    env_info = {
        'platform': sys.platform,
        'python_version': sys.version,
        'is_kaggle': 'KAGGLE_KERNEL_RUN_TYPE' in os.environ,
        'is_colab': 'COLAB_GPU' in os.environ,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    }
    
    print(f"[INFO] ========== 环境信息 ==========")
    print(f"[INFO] 平台: {env_info['platform']}")
    if env_info['is_kaggle']:
        print(f"[INFO] 检测到 Kaggle 环境")
        print(f"[INFO] Kaggle 类型: {os.environ.get('KAGGLE_KERNEL_RUN_TYPE', 'unknown')}")
    elif env_info['is_colab']:
        print(f"[INFO] 检测到 Google Colab 环境")
    else:
        print(f"[INFO] 本地环境")
    
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {env_info['cuda_visible_devices']}")
    return env_info

# ------------------ GPU 详细检测 ------------------
def detailed_gpu_check():
    """详细的GPU检测"""
    print(f"\n[INFO] ========== GPU检测 ==========")
    
    try:
        import torch
        print(f"[INFO] PyTorch版本: {torch.__version__}")
        print(f"[INFO] CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"[INFO] CUDA版本: {torch.version.cuda}")
            gpu_count = torch.cuda.device_count()
            print(f"[INFO] 检测到GPU数量: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                print(f"[INFO] GPU {i}: {props.name} ({memory_gb:.1f}GB)")
                
                # 测试GPU内存
                try:
                    torch.cuda.set_device(i)
                    # 分配小块内存测试
                    test_tensor = torch.zeros(100, 100).cuda()
                    allocated = torch.cuda.memory_allocated(i) / (1024**2)  # MB
                    print(f"[INFO] GPU {i}: 内存测试成功，已分配 {allocated:.1f}MB")
                    del test_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"[WARNING] GPU {i}: 内存测试失败 - {e}")
        else:
            print("[WARNING] CUDA不可用")
            
    except ImportError:
        print("[ERROR] PyTorch未安装")
    except Exception as e:
        print(f"[ERROR] GPU检测失败: {e}")
    
    # ONNX Runtime检测
    try:
        import onnxruntime as ort
        print(f"[INFO] ONNXRuntime版本: {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"[INFO] 可用Providers: {providers}")
    except ImportError:
        print("[ERROR] ONNXRuntime未安装")
    except Exception as e:
        print(f"[ERROR] ONNXRuntime检测失败: {e}")

# ------------------ GPU 设置 ------------------
env_info = detect_environment()
detailed_gpu_check()

if core.globals.use_gpu:
    import torch
    if not torch.cuda.is_available():
        print("[WARNING] GPU不可用，回退到CPU模式")
        core.globals.use_gpu = False
        core.globals.providers = ['CPUExecutionProvider']

# ------------------ 命令行参数 ------------------
import argparse
parser = argparse.ArgumentParser(description="Roop CLI - 多人脸替换（支持多GPU并行处理）")
parser.add_argument('-f', '--face', dest='source_imgs', nargs='+', help='使用的人脸图片列表，可用 skip/none 保留原样')
parser.add_argument('-t', '--target', dest='target_path', required=True, help='目标视频/图片路径')
parser.add_argument('-o', '--output', dest='output_file', help='输出路径')
parser.add_argument('--keep-fps', action='store_true', help='保持原视频帧率')
parser.add_argument('--keep-frames', action='store_true', help='保留中间帧目录')
parser.add_argument('--cores', dest='cores_count', type=int, help='多进程核心数（仅CPU模式）')
parser.add_argument('--from-right', action='store_true', help='从右往左数人脸')
parser.add_argument('--gpu-count', dest='gpu_count', type=int, help='指定使用的GPU数量')
parser.add_argument('--force-single-gpu', action='store_true', help='强制使用单GPU模式')
parser.add_argument('--force-cpu', action='store_true', help='强制使用CPU模式')
parser.add_argument('--debug', action='store_true', help='启用调试模式')

args = parser.parse_args()

# 调试模式
if args.debug:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    print("[INFO] 调试模式已启用")

if not args.source_imgs or len(args.source_imgs) == 0:
    print("[WARNING] 请提供人脸图片列表 -f face1.jpg face2.jpg ...")
    quit()

# ------------------ GPU 设置应用 ------------------
if args.force_cpu:
    print("[INFO] 强制使用CPU模式")
    core.globals.use_gpu = False
    core.globals.use_multi_gpu = False
    core.globals.max_gpu_workers = 1
elif args.force_single_gpu:
    print("[INFO] 强制使用单GPU模式")
    core.globals.use_multi_gpu = False
    core.globals.max_gpu_workers = 1
elif args.gpu_count:
    if args.gpu_count > core.globals.gpu_count:
        print(f"[WARNING] 指定的GPU数量 {args.gpu_count} 超过可用数量 {core.globals.gpu_count}")
        args.gpu_count = core.globals.gpu_count
    core.globals.max_gpu_workers = args.gpu_count
    core.globals.use_multi_gpu = args.gpu_count > 1
    print(f"[INFO] 用户指定使用 {args.gpu_count} 个GPU")

if not args.cores_count:
    args.cores_count = max(1, psutil.cpu_count() - 1)

sep = "/" if os.name != "nt" else "\\"

def start_processing(frame_paths):
    """开始处理视频帧"""
    start_time = time.time()
    total_frames = len(frame_paths)
    
    print(f"\n[INFO] ========== 开始处理 {total_frames} 帧 ==========")
    
    # 显示处理模式信息
    if core.globals.use_gpu and core.globals.use_multi_gpu and core.globals.max_gpu_workers > 1:
        print(f"[INFO] 处理模式: 多GPU并行 ({core.globals.max_gpu_workers} GPU)")
        print(f"[INFO] 每个GPU将在独立进程中运行，实现真正的并行处理")
    elif core.globals.use_gpu:
        print(f"[INFO] 处理模式: 单GPU")
    else:
        print(f"[INFO] 处理模式: CPU ({args.cores_count} 核心)")
    
    try:
        # 调用核心处理函数
        process_video(args.source_imgs, frame_paths)
        
    except KeyboardInterrupt:
        print(f"\n[INFO] 用户中断处理")
        return False
        
    except Exception as e:
        print(f"\n[ERROR] 处理过程中出错: {e}")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        # 尝试回退策略
        if core.globals.use_multi_gpu:
            print("[INFO] 尝试回退到单GPU模式...")
            original_multi_gpu = core.globals.use_multi_gpu
            original_max_workers = core.globals.max_gpu_workers
            
            try:
                core.globals.use_multi_gpu = False
                core.globals.max_gpu_workers = 1
                process_video(args.source_imgs, frame_paths)
            except Exception as e2:
                print(f"[ERROR] 单GPU模式也失败: {e2}")
                if core.globals.use_gpu:
                    print("[INFO] 最后尝试CPU模式...")
                    original_use_gpu = core.globals.use_gpu
                    try:
                        core.globals.use_gpu = False
                        process_video(args.source_imgs, frame_paths)
                        print("[INFO] CPU模式处理成功")
                    except Exception as e3:
                        print(f"[ERROR] 所有模式都失败: {e3}")
                        return False
                    finally:
                        core.globals.use_gpu = original_use_gpu
                else:
                    return False
            finally:
                core.globals.use_multi_gpu = original_multi_gpu
                core.globals.max_gpu_workers = original_max_workers
        else:
            return False
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n[INFO] ========== 处理完成 ==========")
    print(f"[INFO] 总耗时: {elapsed_time:.2f} 秒")
    
    if elapsed_time > 0:
        fps_processed = total_frames / elapsed_time
        print(f"[INFO] 平均处理速度: {fps_processed:.2f} 帧/秒")
        
        # 显示GPU加速效果
        if core.globals.use_multi_gpu and core.globals.max_gpu_workers > 1:
            theoretical_speedup = core.globals.max_gpu_workers
            print(f"[INFO] 多GPU理论加速比: {theoretical_speedup}x")
            print(f"[INFO] 等效单GPU性能: {fps_processed / theoretical_speedup:.2f} 帧/秒")
    
    return True

def main():
    """主函数"""
    print(f"\n[INFO] ========== 开始执行 ==========")
    
    # 检查目标文件
    if not os.path.isfile(args.target_path):
        print(f"[ERROR] 目标文件不存在: {args.target_path}")
        return 1

    # 设置输出文件路径
    if not args.output_file:
        base = os.path.basename(args.target_path)
        if sep in args.target_path:
            args.output_file = rreplace(args.target_path, sep, sep + "swapped-", 1)
        else:
            args.output_file = "swapped-" + base
        print(f"[INFO] 输出文件: {args.output_file}")

    # 测试人脸有效性
    print(f"[INFO] 测试源人脸图片...")
    valid_faces = 0
    for i, src_img in enumerate(args.source_imgs):
        if src_img.lower() not in ["skip", "none"]:
            if os.path.isfile(src_img):
                try:
                    img = cv2.imread(src_img)
                    if img is not None:
                        face = get_face(img)
                        if face is not None:
                            valid_faces += 1
                            print(f"[INFO] 源人脸 {i+1}: ✓ {src_img}")
                        else:
                            print(f"[WARNING] 源人脸 {i+1}: 未检测到人脸 - {src_img}")
                    else:
                        print(f"[WARNING] 源人脸 {i+1}: 无法读取 - {src_img}")
                except Exception as e:
                    print(f"[WARNING] 源人脸 {i+1}: 处理失败 - {src_img} ({e})")
            else:
                print(f"[WARNING] 源人脸 {i+1}: 文件不存在 - {src_img}")
        else:
            print(f"[INFO] 源人脸 {i+1}: 跳过")
    
    if valid_faces == 0:
        print(f"[ERROR] 没有有效的源人脸图片，无法继续处理")
        return 1
    
    print(f"[INFO] 检测到 {valid_faces} 个有效源人脸")

    # 处理图片
    if is_img(args.target_path):
        print(f"[INFO] ========== 图片处理模式 ==========")
        try:
            process_img(args.source_imgs, args.target_path, args.output_file)
            print(f"[INFO] 图片处理成功: {args.output_file}")
            return 0
        except Exception as e:
            print(f"[ERROR] 图片处理失败: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    # 处理视频
    print(f"[INFO] ========== 视频处理模式 ==========")
    video_name = os.path.splitext(os.path.basename(args.target_path))[0]
    output_dir = os.path.join(os.path.dirname(args.target_path), video_name)
    Path(output_dir).mkdir(exist_ok=True)
    print(f"[INFO] 工作目录: {output_dir}")

    # 检测帧率
    fps = detect_fps(args.target_path)
    print(f"[INFO] 原视频帧率: {fps} FPS")
    
    tmp_path = args.target_path
    if not args.keep_fps and fps > 30:
        print(f"[INFO] 将帧率从 {fps} 调整到 30 FPS 以提高处理速度...")
        tmp_path = os.path.join(output_dir, video_name + "_30fps.mp4")
        try:
            set_fps(args.target_path, tmp_path, 30)
            fps = 30
            print(f"[INFO] 帧率调整完成")
        except Exception as e:
            print(f"[WARNING] 帧率调整失败，使用原视频: {e}")
            tmp_path = args.target_path

    print(f"[INFO] 提取视频帧...")
    try:
        # 复制视频文件到工作目录（用于后续音频合成）
        shutil.copy(tmp_path, output_dir)
        
        # 提取帧
        extract_frames(tmp_path, output_dir)
        
        # 获取所有帧文件
        frame_pattern = os.path.join(output_dir, "*.png")
        frame_files = glob.glob(frame_pattern)
        
        if not frame_files:
            print(f"[ERROR] 未找到提取的帧文件")
            return 1
        
        args.frame_paths = tuple(sorted(
            frame_files,
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        ))
        
        print(f"[INFO] 成功提取 {len(args.frame_paths)} 帧")
        
    except Exception as e:
        print(f"[ERROR] 帧提取失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # 处理帧
    if not start_processing(args.frame_paths):
        print(f"[ERROR] 帧处理失败")
        return 1
    
    # 合成视频
    print(f"[INFO] 合成视频...")
    try:
        create_video(video_name, fps, output_dir)
        print(f"[INFO] 视频合成完成")
    except Exception as e:
        print(f"[ERROR] 视频合成失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # 添加音频
    print(f"[INFO] 添加音频...")
    try:
        add_audio(output_dir, tmp_path, args.keep_frames, args.output_file)
        print(f"[INFO] 音频添加完成")
    except Exception as e:
        print(f"[ERROR] 音频添加失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        # 音频添加失败不是致命错误，继续执行
    
    # 显示内存使用情况
    if core.globals.use_gpu and hasattr(core.globals, 'print_gpu_memory_usage'):
        try:
            core.globals.print_gpu_memory_usage()
        except:
            pass
    
    print(f"\n[INFO] ========== 处理完成 ==========")
    print(f"[INFO] 输出视频: {args.output_file}")
    print(f"[INFO] 换脸成功!")
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print(f"\n[INFO] 程序被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {e}")
        if args.debug if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)
