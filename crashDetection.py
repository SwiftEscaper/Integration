import cv2
from ultralytics import YOLO

import getFrame


# 차량 정지 확인 함수
def stop_detection(cars_dict, frame_size, threshold):
    stationary_cars = 0  # 정지한 차량 수
    total_cars = len(cars_dict)  # 전체 차량 수
    
    '''
    with open('txt_file/crash_func.txt', 'a') as file:
        file.write(f'total_cars: {total_cars}\n') 
        
    with open('txt_file/crash_func.txt', 'a') as file:
        file.write(f'cars_dict: {cars_dict}\n') 
    '''
    
    # 차량이 없는 경우 바로 False 반환
    if total_cars == 0:
        return False
    
    # 모든 차량에 대해 사고 여부 확인
    for car_id, data in cars_dict.items():
        
        # data: [[54, 461], [61, 458], [69, 454], [78, 449]]
        
        #with open('center.txt', 'a') as file:
        #    file.write(f'center: {data}\n') 
        
        # 각 차량의 마지막 frame_size 프레임 동안의 x, y 좌표 리스트 가져오기
        centers = data[-frame_size:]
        x_positions = [pos[0] for pos in centers]  # x 좌표 리스트 
        y_positions = [pos[1] for pos in centers]  # y 좌표 리스트
        
        #with open('txt_file/crash_func.txt', 'a') as file:
        #    file.write(f'{centers}\n') 
        
        # x_positions의 길이가 1 이하일 경우 건너뛰기
        if len(x_positions) <= 1:
            continue
        
        # 변화 없는 프레임 수 카운트
        stationary_frames = 0
        #threshold = 2  # 임계값 설정 (예: 2 픽셀 이하의 변화는 정지로 판단)

        # 한 차량에 대해 계산 (100프레임동안)
        for i in range(1, len(x_positions)):
            x_diff = abs(x_positions[i] - x_positions[i-1])
            y_diff = abs(y_positions[i] - y_positions[i-1])
            
            # 변화량이 임계값 이하인 경우 stationary_frames 증가
            if x_diff < threshold and y_diff < threshold:
                stationary_frames += 1
                
            '''
            with open('crash_func.txt', 'a') as file:
                file.write(f'x_diff: {x_diff}, y_diff: {y_diff}\n') 
            '''
                
            #print('stationary_frames: ', stationary_frames)
        
        ################### 이번 차량 계산 종료
        
        # 비율 계산 (100 프레임 동안 한 차량에 대해 50% 이상 멈춰있으면 사고로 간주)
        stationary_ratio = stationary_frames / (len(x_positions) - 1)
        
        # 50% 이상 프레임 동안 멈춰있다면 사고로 판단 (한 차량에서 프레임 중 50% 이상 멈춰있으면)
        if stationary_ratio >= 0.5:
            stationary_cars += 1  # 정지한 차량 수 증가
     
            
    # 정지한 차량이 전체의 50% 이상이면 사고로 판단
    if stationary_cars / total_cars >= 0.5:
        return True
     
    return False


# 각 객체의 중심점(x, y) 좌표를 누적 저장하는 함수 (중심점만 기록)
def update_car_data_xy(cars_dict, bounding_boxes, track_ids):   
    # bounding_boxes: [[1157.3149,  542.3386,  119.2910,  104.2602]]  # x, y, w, h
    for box, track_id in zip(bounding_boxes, track_ids):
        #print(box)
        x, y, w, h = map(int, box)  # 박스 좌표에서 중심점(x, y) 추출
        
        if track_id not in cars_dict:
            cars_dict[track_id] = []  # 새로운 트랙 ID의 경우 빈 리스트 생성
            
        cars_dict[track_id].append([x, y])  # 중심점 좌표를 리스트에 추가

    return cars_dict


# 프레임에 없는 차량 정보 제거 함수 (max_frames_missing동안 없다면)
def remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing):
    # cars_dict에 있는 차량 중 track_ids에 없는 차량 제거 
    for track_id in list(cars_dict.keys()):
        if track_id not in track_ids:  # 차량이 없어지면
            if track_id in frames_since_last_seen:
                frames_since_last_seen[track_id] += 1
            else:
                frames_since_last_seen[track_id] = 1   

            # 조건 검사
            if frames_since_last_seen[track_id] > max_frames_missing:
                del cars_dict[track_id]
                del frames_since_last_seen[track_id] 


def main():
    max_frames_missing = 10  # 이 프레임 이상 없으면 객체 삭제
    frames_since_last_seen = {}
    cars_dict = {}  # id: [x, y] 형식으로 저장
    counter = 1

    cctv_url, processed_name = getFrame.get_cctv_data(LATITUDE, LONGITUDE, CCTV_KEY)
    
    #cctv_url = "trafficAccident1.mp4"
    
    model = YOLO(YOLO_MODEL_PATH)
    video = cv2.VideoCapture(cctv_url)
    
    #cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Video', 600, 400)  # width, height
    
    
    while True:
        print(counter)
        frame, bounding_boxes, track_ids = getFrame.process_frame(video, model)
        
        cars_dict = update_car_data_xy(cars_dict, bounding_boxes, track_ids)
        
        '''
        with open('track.txt', 'a') as file:
            file.write(f'track_ids: {track_ids}\n')
        '''
            
        remove_missing_cars(cars_dict, track_ids, frames_since_last_seen, max_frames_missing)
        
        # 100 frame마다 stop_detection 이 함수 호출에서 추돌 확인
        if counter % 100 == 0:
            flag = stop_detection(cars_dict)
            
            with open('crash_result.txt', 'a') as file:
                file.write(f'result: {flag}\n')
        
        cv2.imshow('Video', frame) # 프레임을 가져와서 이미지에 표시
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        
        counter += 1
        
    # while End
        
# main End


if __name__ == "__main__":
    YOLO_MODEL_PATH = 'yolov8n.pt'
    
    LATITUDE = 37.517423
    LONGITUDE = 127.17903
    
    CCTV_KEY = '19d87e10ec6a47938d779192bd5ef763'
    
    main()