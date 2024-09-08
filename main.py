import cv2
from ultralytics import YOLO
import logging
import requests

import fireDetection
import crashDetection
import getFrame
from accidentEnum import Accident

logging.basicConfig(filename="log.txt", filemode="w", level=logging.DEBUG)

YOLO_MODEL_PATH = 'yolov8n.pt'

LATITUDE = 37.517423
LONGITUDE = 127.17903

CCTV_KEY = '19d87e10ec6a47938d779192bd5ef763'


def main():
    model = YOLO(YOLO_MODEL_PATH)  
    
    cctv_url, cctv_name = getFrame.get_cctv_data(LATITUDE, LONGITUDE, CCTV_KEY)
    video = cv2.VideoCapture(cctv_url)
    
    counter = 1
    
    while True:        
        print(f'Processing frame: {counter}')
        
        frame, bounding_boxes, track_ids = getFrame.process_frame(video, model)
        
        if frame is None:
            logging.warning('Frame is None, stopping video processing')
            break
        
        
        cv2.imshow('Video', frame)
        
        
        ###########################################################################################################
        
        if False:
        #if event.is_set():
            accident_data = {
                'tunnel_name': cctv_name,
                'accident_type': accident_flag.value,
                'latitude': accident_lat.value,
                'longitude': accident_lng.value
            }

            # 서버에 사고 정보 전송
            
            try:
                response = requests.post("http://61.252.59.35:8080/api/accident/", json=accident_data)
                #response = requests.post("http://127.0.0.1:8000//api/accident/", json=accident_data)  # local address
                
                logging.info(f'Server Response: {response.json()} -> server success')
                
            except Exception as e:
                logging.error(f"Error occurred while reporting accident: {e}")
                              
    
            event.clear()  # Reset the event after handling the accident

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter += 1
        
    cv2.destroyAllWindows()
    
# Function End
    

if __name__ == "__main__":
    logging.info('Start')
    main()
    logging.info('End')