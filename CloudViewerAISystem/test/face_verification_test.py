# -*- coding: UTF-8 -*-


import os
import json
import time
import requests
import traceback

# http_url = "http://192.168.17.137:5000/facerecognition?userid=7153"
http_url = "http://47.103.28.225:5000/facerecognition?userid=7153"
html_label_input_box_name = "file"
upload_file_path = os.path.join(os.getcwd(), "face_images/timg.jpg")

if __name__ == '__main__':
    loop_num = 1
    for num in range(loop_num):
        try:
            file_read = open(upload_file_path, 'rb')
            files = {'file': file_read}
            start = time.time()
            http_response = requests.post(http_url, files=files)
            print("#" * 50 + " time span: {} ".format(time.time() - start) + "#" * 50)
            helmet_detect_result_dict = \
                json.loads(http_response.text, encoding="utf-8")

            helmet_detect_result_json = \
                json.dumps(helmet_detect_result_dict, ensure_ascii=False, sort_keys=True, indent=4)
            print(helmet_detect_result_json)
        except Exception as e:
            print(traceback.format_exc())
            continue
