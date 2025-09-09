import onnxruntime
import torch

use_gpu = True
available_providers = onnxruntime.get_available_providers()

if use_gpu and 'CUDAExecutionProvider' in available_providers and torch.cuda.is_available():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    use_gpu = False
    providers = ['CPUExecutionProvider']

print(f"[INFO] Using GPU: {use_gpu}, ONNXRuntime providers: {providers}")
