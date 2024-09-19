import cv2
from ultralytics import YOLO
import logging
import requests

import fireDetection
import crashDetection
import getFrame
import accidentHandler
from accidentEnum import Accident

import time


YOLO_MODEL_PATH = 'yolov8n.pt'

LATITUDE = 37.517423
LONGITUDE = 127.17903

CCTV_KEY = '19d87e10ec6a47938d779192bd5ef763'

logging.basicConfig(filename="log.txt", filemode="w", level=logging.DEBUG)


# 시간 전용 로거
time_logger = logging.getLogger('timeLogger')
time_logger.setLevel(logging.INFO)
time_handler = logging.FileHandler("time_log.txt", mode='w')
time_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
time_handler.setFormatter(formatter)
time_logger.addHandler(time_handler)


def main():
    model = YOLO(YOLO_MODEL_PATH)  
    
    cctv_url, cctv_name = getFrame.get_cctv_data(LATITUDE, LONGITUDE, CCTV_KEY)
    
    #cctv_name = "temp"
    #cctv_url = "input.mp4"
    
    video = cv2.VideoCapture(cctv_url)
    
    counter = 1
    
    max_frames_missing = 10  # 이 프레임 이상 없으면 객체 삭제
    frames_since_last_seen = {}
    cars_dict = {}  # id: [x, y] 형식으로 저장
    frame_size = 100  # 추돌에서 몇 프레임동안 검사할건지
    
    start_time = time.time()  # 시작 시간 기록
    
    while True:        
        print(f'Processing frame: {counter}')
        
        frame, bounding_boxes, track_ids = getFrame.process_frame(video, model)
        
        if frame is None:
            logging.warning('Frame is None, stopping video processing')
            break
        
        ########################################## crash 시작
        #'''
        cars_dict = crashDetection.update_car_data_xy(cars_dict, bounding_boxes, track_ids)
        crashDetection.remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
        
        if counter % frame_size == 0:
            crash_flag = crashDetection.stop_detection(cars_dict, frame_size=frame_size, threshold=2)  # threshold는 몇 픽셀 이하 움직임을 추돌로 판단
            logging.info(f'Crash Result: {crash_flag}')
            
            if crash_flag:
                # 센서 값 get 요청
                # get 응답 받아서 다시 사고 VM으로 정보 전송
                accidentHandler.send_accident_data(cctv_name, Accident.CRASH, LATITUDE, LONGITUDE)
        #'''
        ########################################## crash 끝
        
        ########################################## fire 시작

        
        # Convert bounding boxes to the format required by fire detection
        vehicle_boxes = []
        if bounding_boxes is not None:
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                vehicle_boxes.append([x1, y1, x2, y2])
        
        # Fire detection using fireDetection module
        fire_dict = fireDetection.detect_fire(frame, vehicle_boxes)
        #logging.info(f'Fire Result: {fire_dict}')
        
        if fire_dict:
            for fire in fire_dict:
                #logging.info(f'Fire Result: {fire_dict}')
                
                confidence = fire['confidence']
                
                if confidence >= 0.75:
                    logging.info(f'Fire Result: True, {fire_dict}')
                    
                    x1, y1, x2, y2 = fire['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    fire_class = fire['class']
                    label = f"{fire_class} ({confidence:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # 센서 값 get 요청
                    # get 응답 받아서 다시 사고 VM으로 정보 전송
                    accidentHandler.send_accident_data(cctv_name, Accident.FIRE, LATITUDE, LONGITUDE)
                    
        else:
            logging.info('Fire Result: False')
        
        ########################################## fire 끝

        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        
        # 200 프레임마다 경과 시간 기록
        if counter % 200 == 0:
            end_time = time.time() - start_time
            time_logger.info(f'Processed 200 frames in {end_time:.2f} seconds')
            start_time = time.time()  # 시간 초기화
            
        counter += 1
        
    cv2.destroyAllWindows()
    
# Function End
    

if __name__ == "__main__":
    logging.info('Start')
    main()
    logging.info('End')