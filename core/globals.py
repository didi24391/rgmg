import onnxruntime
import torch
import os

# 基础GPU设置
use_gpu = True
available_providers = onnxruntime.get_available_providers()

# 检测GPU环境
gpu_count = 0
if use_gpu and 'CUDAExecutionProvider' in available_providers:
    try:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[INFO] 检测到 {gpu_count} 个GPU设备")
            
            # 显示GPU详细信息
            for i in range(gpu_count):
                try:
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"[INFO] GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                except Exception as e:
                    print(f"[INFO] GPU {i}: 无法获取详细信息 ({e})")
        else:
            print("[INFO] PyTorch CUDA不可用")
            use_gpu = False
            providers = ['CPUExecutionProvider']
    except Exception as e:
        print(f"[WARNING] GPU检测失败: {e}")
        use_gpu = False
        providers = ['CPUExecutionProvider']
        gpu_count = 0
else:
    use_gpu = False
    providers = ['CPUExecutionProvider']
    print("[INFO] 未检测到CUDA支持，使用CPU模式")

# 多GPU配置策略
max_gpu_workers = 1
use_multi_gpu = False

if gpu_count > 1 and use_gpu:
    # 检查环境变量以确定多GPU策略
    kaggle_gpu = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
    colab_gpu = 'COLAB_GPU' in os.environ
    
    if kaggle_gpu or colab_gpu:
        # Kaggle/Colab环境：更积极的多GPU使用
        max_gpu_workers = gpu_count
        use_multi_gpu = True
        print(f"[INFO] 检测到云环境，启用所有 {gpu_count} 个GPU")
    else:
        # 本地环境：保守策略
        max_gpu_workers = min(gpu_count, 2)
        use_multi_gpu = gpu_count > 1
        print(f"[INFO] 本地环境，使用 {max_gpu_workers} 个GPU")
    
    print(f"[INFO] 多GPU模式启用: {use_multi_gpu}")
else:
    if gpu_count <= 1:
        print("[INFO] 单GPU/无GPU环境，使用单线程模式")
    else:
        print("[INFO] GPU检测失败，回退到单线程模式")

# 显示最终配置
print(f"\n[INFO] === 最终配置 ===")
print(f"[INFO] 使用GPU: {use_gpu}")
print(f"[INFO] GPU数量: {gpu_count}")
print(f"[INFO] 多GPU模式: {use_multi_gpu}")
print(f"[INFO] 最大GPU工作数: {max_gpu_workers}")
print(f"[INFO] ONNXRuntime Providers: {providers}")

# 环境检查和优化建议
if use_gpu:
    try:
        # 检查CUDA版本匹配
        cuda_version = torch.version.cuda
        print(f"[INFO] CUDA版本: {cuda_version}")
        
        # 内存优化设置
        if gpu_count > 0:
            # 为多GPU设置优化参数
            torch.backends.cudnn.benchmark = True  # 优化卷积性能
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 减少内存碎片
            
            if use_multi_gpu:
                # 多GPU特定优化
                os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'  # 使用PCI总线ID排序
                print("[INFO] 已设置多GPU优化参数")
            
    except Exception as e:
        print(f"[WARNING] GPU优化设置失败: {e}")

# 性能监控辅助函数
def get_gpu_memory_info():
    """获取GPU内存使用情况"""
    if not use_gpu or gpu_count == 0:
        return None
    
    memory_info = {}
    try:
        for i in range(gpu_count):
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
                cached = torch.cuda.memory_reserved(i) / (1024**3)  # GB
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
                memory_info[f'gpu_{i}'] = {
                    'allocated': allocated,
                    'cached': cached, 
                    'total': total,
                    'free': total - cached
                }
    except Exception as e:
        print(f"[WARNING] 获取GPU内存信息失败: {e}")
        
    return memory_info

def print_gpu_memory_usage():
    """打印GPU内存使用情况"""
    memory_info = get_gpu_memory_info()
    if memory_info:
        print("\n[INFO] GPU内存使用情况:")
        for gpu_id, info in memory_info.items():
            print(f"  {gpu_id.upper()}: "
                  f"已用 {info['allocated']:.1f}GB / "
                  f"缓存 {info['cached']:.1f}GB / "
                  f"总计 {info['total']:.1f}GB "
                  f"(剩余 {info['free']:.1f}GB)")

# 错误恢复配置
recovery_config = {
    'max_retries': 3,
    'fallback_to_cpu': True,
    'reduce_batch_size_on_oom': True,
    'clear_cache_on_error': True
}
