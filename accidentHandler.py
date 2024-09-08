import logging
import requests

def send_accident_data(cctv_name, accident_type, lat, lng):
    accident_data = {
        'tunnel_name': cctv_name,
        'accident_type': accident_type.value, 
        'latitude': lat,
        'longitude': lng
    }
    
    try:
        response = requests.post("http://61.252.59.35:8080/api/accident/", json=accident_data)
        logging.info(f'Server Response: {response.json()} -> server success')
        
    except Exception as e:
        logging.error(f"Error occurred while reporting accident: {e}")