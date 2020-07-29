# -*- coding: UTF-8 -*-


import os
import json
import requests

# helmet_detect_result_dict = {}
# http_url = "http://192.168.17.150:80/helmetdetector"
# html_label_input_box_name = "file"
# upload_file_path = os.path.join(os.getcwd(),"an-quan-mao-1.jpg")
# if os.path.exists(upload_file_path):
#     c = pycurl.Curl()
#     c.setopt(c.POST, 1)
#     c.setopt(c.URL, http_url)
#     c.setopt(c.HTTPPOST, [(html_label_input_box_name, (c.FORM_FILE, upload_file_path))])
#     c.perform()
#     c.close()

# http_url = 'http://erp.erow.cn:22778/helmetdetector'
http_url = 'http://47.103.28.225:5000/helmetdetector'
# http_url = "http://192.168.17.137:5000/helmetdetector"
# http_url = "http://127.0.0.1:5000/helmetdetector"
# http_url = "http://192.168.17.114:80/helmetdetector"
# upload_file_path = os.path.join(os.getcwd(), "images/199905373241110107.jpg")
upload_file_path = os.path.join(os.getcwd(), "helmet_images/corp_2019_05_01_111.jpg")
file_read = open(upload_file_path, 'rb')
files = {'file': file_read}
ai_images_upload_url_dict = {}
arm_id = '1000'
ai_images_upload_url_dict["pdaNo"] = arm_id
# ai_images_upload_url_dict["fileStatus"] = ''
# ai_images_upload_url_dict["fileDescription"] = file_description
ai_images_upload_url_dict["fileCreateDate"] = '20150211134532'
http_response = requests.post(http_url, params=ai_images_upload_url_dict, files=files)
# print(http_response.text)
# print(type(http_response.text))


helmet_detect_result_dict = \
    json.loads(http_response.text, encoding="utf-8")
helmet_detect_result_json = \
    json.dumps(helmet_detect_result_dict, ensure_ascii=False, sort_keys=True, indent=4)
print(helmet_detect_result_json)

# body_list = []
# helmet_list = []
# match_number = helmet_detect_result_dict["match_number"]
# if match_number > 0:
#     for i in range(match_number):
#         index = i + 1
#         helmet_detect_result_dict["person{0}".format(index)]
