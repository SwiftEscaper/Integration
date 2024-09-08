import numpy as np
import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import sys
import os
from models.common import DetectMultiBackend
from utils.general import (LOGGER, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode

# 경로 및 설정 정의
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # PATH에 ROOT 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로로 변환

# YOLO 모델 로드
device = select_device('')  # GPU
vehicle_model = YOLO("yolov8n.pt")
fire_model = DetectMultiBackend(ROOT / 'best.pt', device=device, dnn=False, data=ROOT / 'data/coco.yaml', fp16=False)  # 화재 탐지 모델

stride, names, pt = fire_model.stride, fire_model.names, fire_model.pt

def process_frame(video, model):
    retval, frame = video.read()  # retval: 성공하면 True
    if not retval:
        return None, None, None  # track_ids 추가

    # 모델 추론
    tracks = model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.2, iou=0.6)
    
    frame = tracks[0].plot()  # track 이미지
    
    bounding_boxes = tracks[0].boxes.xywh.cpu()  # x, y, w, h
    track_ids = tracks[0].boxes.id

    if track_ids is None:
        track_ids = []
    else:
        track_ids = track_ids.int().cpu().tolist()

    return frame, bounding_boxes, track_ids  # 세 개의 값 반환

def calculate_iou(box1, box2):
    """
    두 개의 바운딩 박스 사이의 IoU (Intersection over Union)를 계산하는 함수
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

    return iou

def classify_fire_size(area: int) -> str:
    """화재의 면적에 따라 크기를 분류"""
    if area < 8000:  # 작은 화재(1m)
        return "소형"
    elif 8000 <= area < 20000:  # 중간 크기의 화재 (2.5m)
        return "중형"
    else:  # 큰 화재(3.5m)
        return "대형"

def detect_fire(frame, bounding_boxes, vehicle_boxes):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 이미지를 tensor로 변환하고, 모델이 요구하는 형식으로 변환
    im = torch.from_numpy(frame).permute(2, 0, 1).float().to(fire_model.device)  # HWC -> CHW
    im /= 255.0  # 0-255 범위를 0.0-1.0으로 변환

    if len(im.shape) == 3:
        im = im[None]  # 배치 차원 추가

    # 모델이 요구하는 크기로 이미지 크기 조정
    new_shape = (640, 640)
    im = torch.nn.functional.interpolate(im, size=new_shape, mode='bilinear', align_corners=False)

    # 예측 수행
    pred = fire_model(im, augment=False, visualize=False)
    if isinstance(pred, list):
        pred = pred[0]

    # NMS (Non-Maximum Suppression) 적용
    pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False, max_det=1000)

    detections = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                width, height = x2 - x1, y2 - y1
                area = width * height
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for firec

                # IoU 계산하여 차량 영역과 비교
                ignore_detection = False
                for vehicle_box in vehicle_boxes:
                    iou = calculate_iou([x1, y1, x2, y2], vehicle_box)
                    LOGGER.info(vehicle_box)  # 시작 로그
                    if iou > 0.6:  # 임계값 설정 (예: IoU > 0.5이면 무시)
                        LOGGER.info(f"탐지 결과: 차량 영역과 {iou:.2f} IoU로 겹쳐 무시됨")
                        ignore_detection = True
                        break

                if not ignore_detection:
                    size_class = classify_fire_size(area)
                    detections.append({
                        'bbox': [int(coord) for coord in xyxy],
                        'confidence': float(conf),
                        'class': names[int(cls)],
                        'size' : size_class
                    })
                    # 화재 박스를 프레임에 표시
                    LOGGER.info(f"탐지 결과: {detections[-1]}")
        
        cv2.imshow('Video', frame)

def main():
    LOGGER.info("모델 추론 시작...")  # 시작 로그
    source = str(ROOT / 'input.mp4')  # 로컬 비디오 파일 경로
    
    LOGGER.info(f"비디오 파일 로드 중: {source}")
    vid_cap = cv2.VideoCapture(source)
    
    vehicle_boxes = []

    while True:
        frame, bounding_boxes, track_ids = process_frame(vid_cap, vehicle_model)
        
        if frame is None:
            break
        
        # 차량 박스 저장
        if bounding_boxes is not None:
            vehicle_boxes = []
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                vehicle_boxes.append([x1, y1, x2, y2])
                
        # 화재 탐지
        detect_fire(frame, bounding_boxes, vehicle_boxes)
    
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
