#!/usr/bin/python
# -*- coding: UTF-8 -*-


import os
import shutil


# # create an empty text file
# def create_empty_text_file(folder_path,file_name_without_extention,file_extention):
#     try:
#         file_path = folder_path + file_name_without_extention + file_extention
#         file = open(file_path, 'w')
#         file.close()
#         return True
#     except:
#         return False


# # create a text file and write some message
# def create_text_file(folder_path,file_name_without_extention,file_extention,msg):
#     try:
#         file_path = folder_path + file_name_without_extention + file_extention
#         file = open(file_path, 'w')
#         file.write(msg)
#         file.close()
#         return True
#     except:
#         return False


# python os module rmdir and removedirs can not remove empty dir
# now this dir can remove an empty dir
def remove_dir(dir_path):
    if os.path.isdir(dir_path):
        for p in os.listdir(dir):
            remove_dir(os.path.join(dir_path, p))
        if os.path.exists(dir_path):
            os.rmdir(dir_path)
    else:
        if os.path.exists(dir_path):
            os.remove(dir_path)


# remove dir recursivily
def remove_dir_tree(src_dir_path):
    if os.path.exists(src_dir_path):
        if os.path.isdir(src_dir_path):
            shutil.rmtree(src_dir_path)


# remove a file
def remove_file(src_file_path):
    if not os.path.exists(src_file_path):
        raise Exception("{0}  does not exsit".format(src_file_path))
    os.remove(src_file_path)


# copy src dir to dest dir
# if dest dir path exsits it will raise exception
def copy_dir(src_dir_path, dest_dir_path):
    if not os.path.exists(src_dir_path):
        return
    if os.path.exists(dest_dir_path):
        if os.path.isdir(dest_dir_path):
            shutil.rmtree(dest_dir_path)
    shutil.copytree(src_dir_path, dest_dir_path)


# to ensure src file path exsits
# if src file path does not exsit it will raise exception
# if dest file path exsits we will replace it
def copy_file(src_file_path, dest_file_path):
    if not os.path.exists(src_file_path):
        raise Exception("{0} does not exsit".format(src_file_path))
    shutil.copyfile(src_file_path, dest_file_path)
