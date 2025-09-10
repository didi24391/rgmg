import insightface
import core.globals
import torch

# 全局人脸分析器（主进程使用）
face_analyser = None

def init_face_analyser():
    """初始化人脸分析器（延迟初始化）"""
    global face_analyser
    if face_analyser is None:
        ctx_id = 0 if core.globals.use_gpu else -1
        face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
        face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))
        print("[INFO] 主进程人脸分析器初始化完成")
    return face_analyser

def get_face(img_data, index=0, from_right=False):
    """获取单个人脸（主进程使用）"""
    analyser = init_face_analyser()
    analysed = analyser.get(img_data)
    if not analysed:
        return None

    analysed_sorted = sorted(analysed, key=lambda x: x.bbox[0])
    if from_right:
        analysed_sorted = list(reversed(analysed_sorted))

    if index < len(analysed_sorted):
        return analysed_sorted[index]
    return None

def get_faces(img_data, from_right=False):
    """获取所有人脸（主进程使用）"""
    analyser = init_face_analyser()
    analysed = analyser.get(img_data)
    if not analysed:
        return []

    faces_sorted = sorted(analysed, key=lambda x: x.bbox[0])
    if from_right:
        faces_sorted = list(reversed(faces_sorted))
    return faces_sorted
