import cv2
from ultralytics import YOLO
import logging
import requests

import fireDetection
import crashDetection
import getFrame
import accidentHandler
from accidentEnum import Accident


YOLO_MODEL_PATH = 'yolov8n.pt'

LATITUDE = 37.517423
LONGITUDE = 127.17903

CCTV_KEY = '19d87e10ec6a47938d779192bd5ef763'

logging.basicConfig(filename="log.txt", filemode="w", level=logging.DEBUG)

def main():
    model = YOLO(YOLO_MODEL_PATH)  
    
    cctv_url, cctv_name = getFrame.get_cctv_data(LATITUDE, LONGITUDE, CCTV_KEY)
    video = cv2.VideoCapture(cctv_url)
    
    counter = 1
    
    max_frames_missing = 10  # 이 프레임 이상 없으면 객체 삭제
    frames_since_last_seen = {}
    cars_dict = {}  # id: [x, y] 형식으로 저장
    frame_size = 100  # 추돌에서 몇 프레임동안 검사할건지
    
    while True:        
        print(f'Processing frame: {counter}')
        
        frame, bounding_boxes, track_ids = getFrame.process_frame(video, model)
        
        if frame is None:
            logging.warning('Frame is None, stopping video processing')
            break
        
        ########################################## crash 시작
        
        cars_dict = crashDetection.update_car_data_xy(cars_dict, bounding_boxes, track_ids)
        crashDetection.remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
        
        if counter % frame_size == 0:
            flag = crashDetection.stop_detection(cars_dict, frame_size=frame_size, threshold=2)  # threshold는 몇 픽셀 이하 움직임을 추돌로 판단
            logging.info(f'Crash Result: {flag}')
            
            if flag == Accident.CRASH:
                accidentHandler.send_accident_data(cctv_name, Accident.CRASH, LATITUDE, LONGITUDE)
        
        ########################################## crash 끝
        
        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter += 1
        
    cv2.destroyAllWindows()
    
# Function End
    

if __name__ == "__main__":
    logging.info('Start')
    main()
    logging.info('End')