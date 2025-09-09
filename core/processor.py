import os
import cv2
import insightface
import core.globals
from core.config import get_face, get_faces
from core.utils import rreplace

if os.path.isfile('inswapper_128.onnx'):
    face_swapper = insightface.model_zoo.get_model('inswapper_128.onnx', providers=core.globals.providers)
else:
    quit('File "inswapper_128.onnx" does not exist!')

def process_video(source_imgs, frame_paths):
    source_faces = []
    for src_img in source_imgs:
        if src_img.lower() in ["skip", "none"]:
            source_faces.append(None)
        else:
            img = cv2.imread(src_img)
            face = get_face(img)
            source_faces.append(face if face is not None else None)

    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        faces = get_faces(frame)
        for i, face in enumerate(faces):
            if i < len(source_faces) and source_faces[i] is not None:
                frame = face_swapper.get(frame, face, source_faces[i], paste_back=True)
        cv2.imwrite(frame_path, frame)
        print('.', end='', flush=True)

def process_img(source_imgs, target_path, output_file):
    frame = cv2.imread(target_path)
    faces = get_faces(frame)

    source_faces = []
    for src_img in source_imgs:
        if src_img.lower() in ["skip", "none"]:
            source_faces.append(None)
        else:
            img = cv2.imread(src_img)
            face = get_face(img)
            source_faces.append(face if face is not None else None)

    for i, face in enumerate(faces):
        if i < len(source_faces) and source_faces[i] is not None:
            frame = face_swapper.get(frame, face, source_faces[i], paste_back=True)

    cv2.imwrite(output_file, frame)
    print("\n\nImage saved as:", output_file, "\n\n")
