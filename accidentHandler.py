import logging
import requests

def send_accident_data(cctv_name, accident_type, lat, lng):
    accident_data = {
        'tunnel': cctv_name,
        'type': accident_type.value, 
        'position': 10.0  # temp
    }
    
    try:
        response = requests.post("http://210.102.180.145/api/accident/", json=accident_data)
        logging.info(f'Server Response: {response.json()} -> server success')
        
    except Exception as e:
        logging.error(f"Error occurred while reporting accident: {e}")