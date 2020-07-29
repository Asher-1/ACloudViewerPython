# !/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/9 11:04
# @Author  : ludahai
# @FileName: schedule_tasks.py
# @Software: PyCharm

# !/usr/bin/env python
# coding:utf8

import time
import os
import tarfile
import pickle as p
import shutil
import schedule
import hashlib
from projectinfo import helmets_cache_info


def md5check(fname):
    m = hashlib.md5()
    with open(fname, 'rb') as fobj:
        while True:
            data = fobj.read(4096)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def full_backup(src_dir, dst_dir, md5file):
    par_dir, base_dir = os.path.split(src_dir.rstrip(' / '))
    back_name = '%s_full_%s.tar.gz' % (base_dir, time.strftime('%Y%m%d'))
    full_name = os.path.join(dst_dir, back_name)
    md5dict = {}

    tar = tarfile.open(full_name, 'w:gz')
    arcname = back_name.split(".")[0]
    tar.add(src_dir, arcname=arcname)
    tar.close()
    for path, folders, files in os.walk(src_dir):
        for fname in files:
            full_path = os.path.join(path, fname)
            md5dict[full_path] = md5check(full_path)

    with open(md5file, 'wb') as fobj:
        p.dump(md5dict, fobj)

    # remove src dir and remake src dir
    try:
        shutil.rmtree(src_dir)
        os.makedirs(src_dir)
    except Exception as e:
        print(e)


def incr_backup(src_dir, dst_dir, md5file):
    par_dir, base_dir = os.path.split(src_dir.rstrip('/'))
    back_name = '%s_incr_%s.tar.gz' % (base_dir, time.strftime('%Y%m%d'))
    full_name = os.path.join(dst_dir, back_name)
    md5new = {}

    for path, folders, files in os.walk(src_dir):
        for fname in files:
            full_path = os.path.join(path, fname)
            md5new[full_path] = md5check(full_path)

    if os.path.exists(md5file):
        with open(md5file, 'rb') as fobj:
            md5old = p.load(fobj)
            tar = tarfile.open(full_name, 'w:gz')
            for key in md5new:
                if md5old.get(key) != md5new[key]:
                    arcname = back_name.split(".")[0]
                    tar.add(key, arcname=os.path.join(arcname, os.path.basename(key)))
            tar.close()

    with open(md5file, 'wb') as fobj:
        p.dump(md5new, fobj)


def job(src_dir, dst_dir, md5file):
    if time.strftime('%a') == 'Mon':
        full_backup(src_dir, dst_dir, md5file)
        print("full backup finished...")
    else:
        incr_backup(src_dir, dst_dir, md5file)
        print("incremental backup finished...")


if __name__ == '__main__':
    src_dir = helmets_cache_info.HELMETS_CACHE_IMG_PATH
    dst_dir = helmets_cache_info.HELMETS_CACHE_DIR_PATH
    md5file = os.path.join(dst_dir, "md5.data")

    schedule.every(10).seconds.do(job, src_dir, dst_dir, md5file)
    # schedule.every().day.at("06:30").do(job, src_dir, dst_dir, md5file)
    print("start schedule task ...")
    try:
        while True:
            schedule.run_pending()
            time.sleep(10)
    except Exception as e:
        print(e)
