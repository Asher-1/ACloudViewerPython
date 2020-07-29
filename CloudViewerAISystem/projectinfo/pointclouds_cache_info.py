# -*- coding: UTF-8 -*-


import os

'''
# the current file path and dir path
'''
__current_python_file_path = os.path.realpath(__file__)
__current_python_file_dir_path = os.path.dirname(__current_python_file_path)

'''
# the project root path and dir path
'''
PROJECT_ROOT_PATH = os.path.dirname(__current_python_file_dir_path)
PROJECT_ROOT_DIR_PATH = os.path.dirname(PROJECT_ROOT_PATH)

'''
# ##############################################################
#       aiserverconfigures
# ##############################################################
'''
PROJECT_CONFIGURES_DIR_NAME = "aiserverconfigures"
PROJECT_CONFIGURES_DIR_PATH = os.path.join(PROJECT_ROOT_DIR_PATH, PROJECT_CONFIGURES_DIR_NAME)

'''
# ##############################################################
#       aiserverconfigures ---- pointclouds-cache
# ##############################################################
'''
CLOUDS_CACHE_DIR_NAME = "pointclouds-cache"
CLOUDS_CACHE_DIR_NAME = os.path.join(PROJECT_CONFIGURES_DIR_PATH, CLOUDS_CACHE_DIR_NAME)
IMAGE_CACHE_DIR_NAME = "images-cache"
IMAGE_CACHE_DIR_PATH = os.path.join(PROJECT_CONFIGURES_DIR_PATH, IMAGE_CACHE_DIR_NAME)

# print the paths for the project
if __name__ == "__main__":
    print("========================base==================================")
    print(PROJECT_ROOT_PATH)
    print(PROJECT_ROOT_DIR_PATH)
    print(PROJECT_CONFIGURES_DIR_PATH)
    print(CLOUDS_CACHE_DIR_NAME)
    print(IMAGE_CACHE_DIR_PATH)
