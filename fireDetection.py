import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import sys
import os
from models.common import DetectMultiBackend
from utils.general import (LOGGER, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device, smart_inference_mode

import logging
import getFrame

YOLO_MODEL_PATH = 'yolov8n.pt'

logging.basicConfig(filename="log.txt", filemode="w", level=logging.DEBUG)

# 경로 및 설정 정의
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO 루트 디렉토리
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # PATH에 ROOT 추가
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 상대 경로로 변환

# YOLO 모델 로드
device = select_device('')  # GPU
fire_model = DetectMultiBackend(ROOT / 'best.pt', device=device, dnn=False, data=ROOT / 'data/coco.yaml', fp16=False)  # 화재 탐지 모델

stride, names, pt = fire_model.stride, fire_model.names, fire_model.pt

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
        return "small"
    elif 8000 <= area < 20000:  # 중간 크기의 화재 (2.5m)
        return "medium "
    else:  # 큰 화재(3.5m)
        return "large"


def detect_fire(frame, vehicle_boxes):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 이미지를 tensor로 변환하고, 모델이 요구하는 형식으로 변환
    im = torch.from_numpy(frame).permute(2, 0, 1).float().to(fire_model.device)  # HWC -> CHW
    im /= 255.0  # 0-255 범위를 0.0-1.0으로 변환

    # 배치 차원 추가
    if len(im.shape) == 3:
        im = im[None]  

    # 모델이 요구하는 크기로 이미지 크기 조정
    new_shape = (640, 640)
    im = torch.nn.functional.interpolate(im, size=new_shape, mode='bilinear', align_corners=False)

    # 예측 수행 (첫 번째 결과만 사용)
    pred = fire_model(im, augment=False, visualize=False)
    if isinstance(pred, list):
        pred = pred[0]

    # NMS (Non-Maximum Suppression) 적용 -> 중복되는 바운딩 박스 제거
    # conf = 0.3 이상인 박스만 남김, iou = 0.5 이상인 박스를 제거
    pred = non_max_suppression(pred, 0.3, 0.5, classes=None, agnostic=False, max_det=1000)

    detections = []
    
    for i, det in enumerate(pred):
        if len(det):  # 해당 프레임에서 탐지된 화재 객체가 있다면
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
            
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = map(int, xyxy)
                width, height = x2 - x1, y2 - y1
                area = width * height
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle for fire

                # 각 객체마다 ignore_detection을 False로 초기화
                ignore_detection = False  
            
                # IoU 계산하여 차량 영역과 비교
                for vehicle_box in vehicle_boxes:
                    iou = calculate_iou([x1, y1, x2, y2], vehicle_box)
                    if iou > 0.6:  # 임계값 설정 (예: IoU > 0.5이면 무시)
                        logging.info(f"Fire Result in firedetection.py: Car {iou:.2f} IoU Ignore, flag: {ignore_detection}")
                        ignore_detection = True  
                        continue  # 현재 탐지 무시

                if not ignore_detection:
                    detections.append({
                        'class': names[int(cls)],   # x1, y1, x2, y2
                        'bbox': [int(coord) for coord in xyxy],
                        'confidence': float(conf)
                    })
                    logging.info(f"Fire Result in Func: {detections[-1]}")
        
        #cv2.imshow('Video', frame)
        
    return detections

def main():
    source = str(ROOT / 'fire.mp4')  # 로컬 비디오 파일 경로
    vid_cap = cv2.VideoCapture(source)
    
    '''
    cctv_url, cctv_name = getFrame.get_cctv_data(LATITUDE, LONGITUDE, CCTV_KEY)
    vid_cap = cv2.VideoCapture(cctv_url)
    '''
    
    vehicle_boxes = []

    while True:
        frame, bounding_boxes, track_ids = getFrame.process_frame(vid_cap, YOLO(YOLO_MODEL_PATH))
        
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
        detect_fire(frame, vehicle_boxes)
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    LATITUDE = 37.517423
    LONGITUDE = 127.17903
    
    CCTV_KEY = '19d87e10ec6a47938d779192bd5ef763'
    
    main()
