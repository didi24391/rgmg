import insightface
import core.globals

ctx_id = 0 if core.globals.use_gpu else -1

# 初始化人脸检测
face_analyser = insightface.app.FaceAnalysis(name='buffalo_l', providers=core.globals.providers)
face_analyser.prepare(ctx_id=ctx_id, det_size=(640, 640))

def get_face(img_data, index=0, from_right=False):
    analysed = face_analyser.get(img_data)
    if not analysed:
        return None

    analysed_sorted = sorted(analysed, key=lambda x: x.bbox[0])
    if from_right:
        analysed_sorted = list(reversed(analysed_sorted))

    if index < len(analysed_sorted):
        return analysed_sorted[index]
    return None

def get_faces(img_data, from_right=False):
    analysed = face_analyser.get(img_data)
    if not analysed:
        return []

    faces_sorted = sorted(analysed, key=lambda x: x.bbox[0])
    if from_right:
        faces_sorted = list(reversed(faces_sorted))
    return faces_sorted
