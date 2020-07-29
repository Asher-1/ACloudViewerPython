# -*- coding: UTF-8 -*-

debug_mode = True  # True: failed
if not debug_mode:
    from gevent import monkey
    from gevent.pywsgi import WSGIServer

    monkey.patch_all()

import os
import sys
import json
import datetime
import random
import logging
import shutil
import errno
import requests
from flask import Flask, request
from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse
from werkzeug import secure_filename

work_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(work_dir)

current_file_path = os.path.realpath(__file__)
current_file_dir_path = os.path.dirname(current_file_path)
face_utils_dir_path = os.path.join(current_file_dir_path, "face_utils")

sys.path.append(current_file_dir_path)
sys.path.append(face_utils_dir_path)

print(current_file_dir_path)
print(face_utils_dir_path)

from projectinfo import pointclouds_cache_info
from projectinfo import datasets_url_info
from SemanticSegmentationSystem.ai_point_cloud import AIPointCloud

# the flask app
app = Flask(__name__)

gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# the settings for uploading
if not os.path.exists(pointclouds_cache_info.CLOUDS_CACHE_DIR_NAME):
    os.makedirs(pointclouds_cache_info.CLOUDS_CACHE_DIR_NAME)
if not os.path.exists(pointclouds_cache_info.IMAGE_CACHE_DIR_PATH):
    os.makedirs(pointclouds_cache_info.IMAGE_CACHE_DIR_PATH)
app.config['CLOUDS_CACHE_FOLDER'] = pointclouds_cache_info.CLOUDS_CACHE_DIR_NAME
app.config['IMAGE_CACHE_FOLDER'] = pointclouds_cache_info.IMAGE_CACHE_DIR_PATH
app.config['DATA_URL'] = datasets_url_info.URL

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
ALLOWED_EXTENSIONS = {'vtk', 'jpg', 'jpeg', 'gif'}


# define the allowed file for uploading
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def check_image_size(image_plt, mini_length=500):
    min_side_length = min(image_plt.size)
    if min_side_length < mini_length:
        app.logger.warning("detect invalid image size: {}".format(min_side_length))
        return False
    else:
        return True


# generate a unique id by datetime now
def generate_unique_id():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    random_number = random.randint(0, 100)
    random_number_str = str(random_number)
    if random_number < 10:
        random_number_str = str(0) + str(random_number)
    now_random_str = now_str + "-" + random_number_str
    return now_random_str


def empty_dir(dirname, parent=False):
    def mkdir_p(dir):
        """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
        Args:
            dirname(str):
        """
        assert dir is not None
        if dir == '' or os.path.isdir(dir):
            return
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise e

    try:
        # shutil.rmtree(dirname, ignore_errors=True)
        shutil.rmtree(dirname, ignore_errors=False)
    except Exception as e:
        print(e)
    finally:
        if not parent:
            mkdir_p(dirname)


def get_dir_size(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size / 1024 / 1024 / 1024  # GB


def prepare_data(file_list, cache_dir):
    new_file_list = []
    for file in file_list:
        cache_file = os.path.join(cache_dir, os.path.basename(file))
        if not os.path.exists(file) and not os.path.exists(cache_file):  # // cannot find file in local and cache dir
            url = app.config['DATA_URL'] + os.path.basename(file)
            app.logger.info("downloading file from {} ...".format(url))
            r = requests.get(url)
            with open(cache_file, "wb") as f:
                f.write(r.content)
                new_file_list.append(cache_file)
                app.logger.info("cache file to {}".format(cache_file))
        elif os.path.exists(cache_file):
            new_file_list.append(cache_file)
        elif os.path.exists(file):
            new_file_list.append(file)
    return new_file_list


def get_file_list_time_sorted(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        # os.path.getmtime() last modify time
        # os.path.getctime() last create time
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        # print(dir_list)
        return dir_list


def clear_cache_if_necessary(cache_dir):
    file_size = get_dir_size(cache_dir)
    if file_size > 20:  # > 20GB
        app.logger.info("detect cache size is more than 20 GB, and will remove the oldest history cache.")
        file_list = get_file_list_time_sorted(cache_dir)
        if len(file_list) > 0:
            data = os.path.join(cache_dir, file_list[0])
            if os.path.isdir(data):
                empty_dir(data)
                app.logger.info("remove directory {} ......".format(data))
            elif os.path.isfile(data):
                os.remove(data)
                app.logger.info("remove file {} ......".format(data))


# restful api
api = Api(app)

cloud_ai = AIPointCloud()


# default root url path
class Hello(Resource):
    def get(self):
        return {"message": "Hello ErowCloudViewer AI System!"}


class AICloud(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('need_sample', type=str, location='args', required=False, default='None')

    def post(self):
        url_params = self.parser.parse_args()
        need_sample = url_params.get('need_sample')
        if isinstance(request.json, dict):
            request_info = request.json
        elif isinstance(request.json, str):
            request_info = json.loads(request.json)
        else:
            app.logger.warning("invalid parameters")
            return {'result': [], 'time_take': 0, 'state': "invalid parameters"}

        target_info_list = []
        scene_list = []
        for scene in request_info:
            if "files" not in scene.keys():
                message = "invalid parameters: no files found!"
                app.logger.warning(message)
                return {'result': [], 'time_take': 0, 'state': message}
            if "strategy" not in scene.keys():
                message = "invalid parameters: no strategy found!"
                app.logger.warning(message)
                return {'result': [], 'time_take': 0, 'state': message}
            if "targets" not in scene["strategy"]:
                message = "invalid parameters: no targets found!"
                app.logger.warning(message)
                return {'result': [], 'time_take': 0, 'state': message}

            file_list = scene["files"]
            file_list = prepare_data(file_list, app.config['CLOUDS_CACHE_FOLDER'])
            scene_list.append(file_list)
            target_info = scene["strategy"]
            target_info_list.append(target_info)
        res = cloud_ai.semantic_segmentation(scene_list, target_info_list=target_info_list)
        clear_cache_if_necessary(app.config['CLOUDS_CACHE_FOLDER'])
        return res


api.add_resource(Hello, '/')
api.add_resource(AICloud, '/aiCloud')

if __name__ == '__main__':
    if not debug_mode:
        http_server = WSGIServer(('0.0.0.0', 9995), app)
        http_server.serve_forever()
    else:
        app.run(host='0.0.0.0', port='9995', debug=True)
