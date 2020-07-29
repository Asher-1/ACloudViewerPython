# -*- coding: UTF-8 -*-


import os
import json
import time
import requests
import traceback
from aiserver.face_utils.help_utils import visualize_img

# http_url = "http://192.168.10.111:5000/facerecognition?groupname=启动三&detectAngle=True"
# http_url = "http://47.103.28.225:5000/facerecognition?groupname=启东四"
http_url = "http://58.246.24.158:9997/facerecognition?groupname=启东四"
# http_url = "http://erp.erow.cn:22778/facerecognition?groupname=ckb"
html_label_input_box_name = "file"
# upload_file_path = os.path.join(os.getcwd(), "500115191669631542.jpg")
upload_file_path = os.path.join(os.getcwd(), "face_images/test20200402.jpg")
# upload_file_path = os.path.join(os.getcwd(), "face_images/test220200324.jpg")
# upload_file_path = os.path.join(os.getcwd(), "face_images/test20200326122254.jpg")
# upload_file_path = os.path.join(os.getcwd(), "face_images/847191434656120392.jpg")
result_file_path = os.path.join(os.getcwd(), "test_results")
# upload_file_path = os.path.join(os.getcwd(), "zhan-ban-hui-1.jpg")


if __name__ == '__main__':
    loop_num = 1
    for num in range(loop_num):
        try:
            file_read = open(upload_file_path, 'rb')
            files = {'file': file_read}
            start = time.time()
            http_response = requests.post(http_url, files=files)
            face_detect_result_dict = \
                json.loads(http_response.text, encoding="utf-8")

            res_dict = face_detect_result_dict[0]

            # visualize_img(upload_file_path, res_dict, result_file_path, only_bbox=True)
            visualize_img(upload_file_path, res_dict, result_file_path, only_bbox=False)

            helmet_detect_result_json = \
                json.dumps(face_detect_result_dict, ensure_ascii=False, sort_keys=True, indent=4)
            print(helmet_detect_result_json)
            print("{}: {}\t {}: {}".format("total_number", int(res_dict["total_number"]),
                                           "rotate_angle", int(res_dict["rotate_angle"])))
            print("#" * 50 + " time span: {} ".format(time.time() - start) + "#" * 50)
        except Exception as e:
            print(traceback.format_exc())
            continue
