import os
import cv2
import insightface
import torch
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import core.globals
from core.config import get_face, get_faces

# 全局模型存储
_gpu_models = {}
_model_lock = threading.Lock()

class GPUWorker:
    """GPU工作线程类"""
    def __init__(self, gpu_id, source_faces):
        self.gpu_id = gpu_id
        self.source_faces = source_faces
        self.processed_count = 0
        self.failed_count = 0
        self.models = None
        self._lock = threading.Lock()
        
    def initialize_models(self):
        """初始化GPU模型"""
        if self.models is not None:
            return True
            
        try:
            print(f"[GPU {self.gpu_id}] 初始化模型...")
            
            # 设置CUDA设备
            torch.cuda.set_device(self.gpu_id)
            
            # 创建ONNX provider配置
            if torch.cuda.is_available() and self.gpu_id < torch.cuda.device_count():
                providers = [
                    ('CUDAExecutionProvider', {'device_id': self.gpu_id}),
                    'CPUExecutionProvider'
                ]
                ctx_id = self.gpu_id
                print(f"[GPU {self.gpu_id}] 使用CUDA Provider")
            else:
                providers = ['CPUExecutionProvider']
                ctx_id = -1
                print(f"[GPU {self.gpu_id}] 回退到CPU Provider")
            
            # 检查模型文件
            if not os.path.isfile('inswapper_128.onnx'):
                raise FileNotFoundError('inswapper_128.onnx not found')
            
            # 创建人脸交换器
            face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=providers)
            
            # 创建人脸分析器
            face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=providers)
            face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))
            
            self.models = {
                'swapper': face_swapper,
                'analyser': face_analyser
            }
            
            print(f"[GPU {self.gpu_id}] 模型初始化完成")
            return True
            
        except Exception as e:
            print(f"[GPU {self.gpu_id}] 模型初始化失败: {e}")
            return False
    
    def process_frame(self, frame_path):
        """处理单个帧"""
        if self.models is None:
            if not self.initialize_models():
                return False, "模型初始化失败"
        
        try:
            # 读取帧
            frame = cv2.imread(frame_path)
            if frame is None:
                return False, f"无法读取帧: {frame_path}"
            
            # 检测人脸
            faces = self.models['analyser'].get(frame)
            if not faces:
                # 没有人脸也要保存原帧
                cv2.imwrite(frame_path, frame)
                return True, "无人脸"
            
            # 排序人脸
            faces_sorted = sorted(faces, key=lambda x: x.bbox[0])
            
            # 替换人脸
            swap_count = 0
            for i, face in enumerate(faces_sorted):
                if i < len(self.source_faces) and self.source_faces[i] is not None:
                    try:
                        frame = self.models['swapper'].get(frame, face, self.source_faces[i], paste_back=True)
                        swap_count += 1
                    except Exception as e:
                        print(f"[GPU {self.gpu_id}] 人脸替换失败: {e}")
                        continue
            
            # 保存帧
            cv2.imwrite(frame_path, frame)
            
            with self._lock:
                self.processed_count += 1
                
            return True, f"成功替换{swap_count}张人脸"
            
        except Exception as e:
            with self._lock:
                self.failed_count += 1
            return False, f"处理失败: {e}"
    
    def get_stats(self):
        """获取统计信息"""
        with self._lock:
            return {
                'processed': self.processed_count,
                'failed': self.failed_count,
                'total': self.processed_count + self.failed_count
            }

class MultiGPUManager:
    """多GPU管理器"""
    def __init__(self, gpu_count, source_faces):
        self.gpu_count = gpu_count
        self.source_faces = source_faces
        self.workers = []
        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        # 创建工作器
        for gpu_id in range(gpu_count):
            worker = GPUWorker(gpu_id, source_faces)
            self.workers.append(worker)
    
    def worker_thread(self, worker):
        """工作线程函数"""
        thread_id = threading.current_thread().ident
        print(f"[GPU {worker.gpu_id}] 工作线程启动 (Thread ID: {thread_id})")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取帧路径
                frame_path = self.frame_queue.get(timeout=1.0)
                if frame_path is None:  # 结束信号
                    break
                
                # 处理帧
                success, message = worker.process_frame(frame_path)
                
                # 将结果放入结果队列
                self.result_queue.put({
                    'gpu_id': worker.gpu_id,
                    'frame_path': frame_path,
                    'success': success,
                    'message': message,
                    'thread_id': thread_id
                })
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[GPU {worker.gpu_id}] 工作线程异常: {e}")
                continue
        
        print(f"[GPU {worker.gpu_id}] 工作线程结束")
    
    def process_frames(self, frame_paths, progress_callback=None):
        """处理所有帧"""
        total_frames = len(frame_paths)
        print(f"[INFO] 使用 {self.gpu_count} 个GPU处理 {total_frames} 帧")
        
        # 将所有帧路径放入队列
        for frame_path in frame_paths:
            self.frame_queue.put(frame_path)
        
        # 启动工作线程
        threads = []
        for worker in self.workers:
            thread = threading.Thread(target=self.worker_thread, args=(worker,))
            thread.start()
            threads.append(thread)
        
        # 监控进度
        start_time = time.time()
        processed_frames = 0
        failed_frames = 0
        
        # 收集结果
        results_collected = 0
        while results_collected < total_frames:
            try:
                result = self.result_queue.get(timeout=5.0)
                results_collected += 1
                
                if result['success']:
                    processed_frames += 1
                else:
                    failed_frames += 1
                    if failed_frames <= 5:  # 只显示前5个错误
                        print(f"[WARNING] {result['message']}")
                
                # 定期报告进度
                if results_collected % 50 == 0 or results_collected == total_frames:
                    elapsed = time.time() - start_time
                    progress = (results_collected / total_frames) * 100
                    if elapsed > 0:
                        fps = results_collected / elapsed
                        eta = (total_frames - results_collected) / fps if fps > 0 else 0
                        print(f"[INFO] 进度: {results_collected}/{total_frames} ({progress:.1f}%) "
                              f"- {fps:.1f} fps - ETA: {eta:.1f}s")
                
                if progress_callback:
                    progress_callback(results_collected, total_frames)
                
            except queue.Empty:
                print("[WARNING] 结果队列超时，检查工作线程状态...")
                # 检查线程是否还活着
                alive_threads = [t for t in threads if t.is_alive()]
                if not alive_threads:
                    print("[ERROR] 所有工作线程都已结束")
                    break
                continue
        
        # 停止所有线程
        self.stop_event.set()
        
        # 发送结束信号
        for _ in self.workers:
            self.frame_queue.put(None)
        
        # 等待所有线程结束
        for thread in threads:
            thread.join(timeout=5.0)
        
        elapsed_time = time.time() - start_time
        
        # 收集统计信息
        total_processed = sum(w.processed_count for w in self.workers)
        total_failed = sum(w.failed_count for w in self.workers)
        
        print(f"\n[INFO] === 多GPU处理完成 ===")
        print(f"[INFO] 总帧数: {total_frames}")
        print(f"[INFO] 成功处理: {total_processed}")
        print(f"[INFO] 失败: {total_failed}")
        print(f"[INFO] 成功率: {(total_processed/total_frames)*100:.1f}%")
        print(f"[INFO] 总耗时: {elapsed_time:.2f} 秒")
        
        if elapsed_time > 0:
            avg_fps = total_processed / elapsed_time
            print(f"[INFO] 平均速度: {avg_fps:.2f} 帧/秒")
            
            # 估算加速比
            single_gpu_fps = avg_fps / self.gpu_count
            print(f"[INFO] 理论加速比: {self.gpu_count}x")
            print(f"[INFO] 等效单GPU速度: {single_gpu_fps:.2f} 帧/秒")
        
        # 显示各GPU统计
        print(f"\n[INFO] 各GPU处理统计:")
        for i, worker in enumerate(self.workers):
            stats = worker.get_stats()
            success_rate = (stats['processed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  GPU {i}: 处理 {stats['processed']}, 失败 {stats['failed']}, 成功率 {success_rate:.1f}%")

def prepare_source_faces(source_imgs):
    """准备源人脸数据"""
    print("[INFO] 准备源人脸数据...")
    source_faces = []
    
    for i, src_img in enumerate(source_imgs):
        if src_img.lower() in ["skip", "none"]:
            source_faces.append(None)
            print(f"[INFO] 源人脸 {i+1}: 跳过")
        else:
            try:
                img = cv2.imread(src_img)
                if img is not None:
                    face = get_face(img)
                    source_faces.append(face)
                    if face is not None:
                        print(f"[INFO] 源人脸 {i+1}: 成功加载 {src_img}")
                    else:
                        print(f"[WARNING] 源人脸 {i+1}: 未检测到人脸 {src_img}")
                else:
                    source_faces.append(None)
                    print(f"[WARNING] 源人脸 {i+1}: 无法读取 {src_img}")
            except Exception as e:
                source_faces.append(None)
                print(f"[WARNING] 源人脸 {i+1} 处理失败: {e}")
    
    valid_faces = sum(1 for face in source_faces if face is not None)
    print(f"[INFO] 成功加载 {valid_faces} 个有效源人脸")
    return source_faces

def process_video_multi_gpu_threaded(source_imgs, frame_paths):
    """基于线程的多GPU处理"""
    gpu_count = core.globals.max_gpu_workers
    
    print(f"\n[INFO] === 多GPU线程模式 ===")
    print(f"[INFO] 使用GPU数量: {gpu_count}")
    print(f"[INFO] 总帧数: {len(frame_paths)}")
    
    # 准备源人脸
    source_faces = prepare_source_faces(source_imgs)
    if sum(1 for f in source_faces if f is not None) == 0:
        print("[WARNING] 没有有效的源人脸")
        return
    
    # 创建多GPU管理器
    manager = MultiGPUManager(gpu_count, source_faces)
    
    try:
        # 处理帧
        manager.process_frames(frame_paths)
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断处理")
        manager.stop_event.set()
    except Exception as e:
        print(f"[ERROR] 多GPU处理失败: {e}")
        raise

def process_video_single_gpu(source_imgs, frame_paths):
    """单GPU处理"""
    device_name = "GPU" if core.globals.use_gpu else "CPU"
    print(f"\n[INFO] === 单{device_name}模式 ===")
    print(f"[INFO] 总帧数: {len(frame_paths)}")
    
    # 准备源人脸
    source_faces = prepare_source_faces(source_imgs)
    if sum(1 for f in source_faces if f is not None) == 0:
        print("[WARNING] 没有有效的源人脸")
        return
    
    # 创建单GPU工作器
    gpu_id = 0 if core.globals.use_gpu else -1
    worker = GPUWorker(gpu_id, source_faces)
    
    if not worker.initialize_models():
        print("[ERROR] 模型初始化失败")
        return
    
    # 处理帧
    start_time = time.time()
    
    for i, frame_path in enumerate(frame_paths):
        success, message = worker.process_frame(frame_path)
        
        if not success and worker.failed_count <= 5:
            print(f"[WARNING] {message}")
        
        # 定期显示进度
        if (i + 1) % 100 == 0 or (i + 1) == len(frame_paths):
            progress = ((i + 1) / len(frame_paths)) * 100
            elapsed = time.time() - start_time
            if elapsed > 0:
                fps = (i + 1) / elapsed
                eta = (len(frame_paths) - i - 1) / fps if fps > 0 else 0
                print(f"[INFO] 进度: {i+1}/{len(frame_paths)} ({progress:.1f}%) "
                      f"- {fps:.1f} fps - ETA: {eta:.1f}s")
    
    elapsed_time = time.time() - start_time
    stats = worker.get_stats()
    
    print(f"\n[INFO] === 单{device_name}处理完成 ===")
    print(f"[INFO] 成功: {stats['processed']}, 失败: {stats['failed']}")
    print(f"[INFO] 成功率: {(stats['processed']/len(frame_paths)*100):.1f}%")
    print(f"[INFO] 耗时: {elapsed_time:.2f} 秒")
    if elapsed_time > 0:
        print(f"[INFO] 平均速度: {stats['processed']/elapsed_time:.2f} 帧/秒")

def process_video(source_imgs, frame_paths):
    """主处理函数"""
    total_frames = len(frame_paths)
    
    print(f"\n[INFO] ========== 视频处理开始 ==========")
    print(f"[INFO] 总帧数: {total_frames}")
    print(f"[INFO] GPU模式: {core.globals.use_gpu}")
    if core.globals.use_gpu:
        print(f"[INFO] GPU数量: {core.globals.gpu_count}")
        print(f"[INFO] 多GPU模式: {core.globals.use_multi_gpu}")
        print(f"[INFO] 使用GPU数: {core.globals.max_gpu_workers}")
    
    # 选择处理模式
    use_multi_gpu = (
        core.globals.use_gpu and 
        core.globals.use_multi_gpu and 
        core.globals.max_gpu_workers > 1 and
        total_frames >= 50
    )
    
    try:
        if use_multi_gpu:
            print(f"[INFO] 使用多GPU线程模式 ({core.globals.max_gpu_workers} GPU)")
            process_video_multi_gpu_threaded(source_imgs, frame_paths)
        else:
            mode = "单GPU" if core.globals.use_gpu else "CPU"
            print(f"[INFO] 使用{mode}模式")
            process_video_single_gpu(source_imgs, frame_paths)
    
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 自动回退
        if use_multi_gpu:
            print("[INFO] 尝试回退到单GPU...")
            try:
                process_video_single_gpu(source_imgs, frame_paths)
            except Exception as e2:
                print(f"[ERROR] 单GPU也失败: {e2}")
                if core.globals.use_gpu:
                    print("[INFO] 最后尝试CPU模式...")
                    original_use_gpu = core.globals.use_gpu
                    try:
                        core.globals.use_gpu = False
                        process_video_single_gpu(source_imgs, frame_paths)
                    finally:
                        core.globals.use_gpu = original_use_gpu
                else:
                    raise e2
        else:
            raise e
    
    print("[INFO] ========== 视频处理结束 ==========\n")

def process_img(source_imgs, target_path, output_file):
    """处理单张图片"""
    print("[INFO] 开始图片处理...")
    
    try:
        frame = cv2.imread(target_path)
        if frame is None:
            print(f"[ERROR] 无法读取目标图片: {target_path}")
            return
        
        faces = get_faces(frame)
        print(f"[INFO] 检测到 {len(faces)} 张人脸")
        
        # 准备源人脸
        source_faces = prepare_source_faces(source_imgs)
        
        # 使用单GPU工作器处理
        worker = GPUWorker(0 if core.globals.use_gpu else -1, source_faces)
        if not worker.initialize_models():
            print("[ERROR] 模型初始化失败")
            return
        
        # 临时保存图片进行处理
        temp_path = target_path + ".temp.png"
        cv2.imwrite(temp_path, frame)
        
        success, message = worker.process_frame(temp_path)
        
        if success:
            # 读取处理后的图片
            result_frame = cv2.imread(temp_path)
            if result_frame is not None:
                cv2.imwrite(output_file, result_frame)
                print(f"[INFO] 图片处理成功: {output_file}")
            else:
                print("[ERROR] 无法读取处理后的图片")
        else:
            print(f"[ERROR] 图片处理失败: {message}")
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    except Exception as e:
        print(f"[ERROR] 图片处理失败: {e}")
        raise
