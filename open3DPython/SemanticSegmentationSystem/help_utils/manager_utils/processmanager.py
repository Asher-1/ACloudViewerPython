#!/usr/bin/python
# -*- coding: UTF-8 -*-


import psutil




# get the process name by process id
def get_process_name_by_id(process_id):
    process_name = ''
    pids = psutil.pids()
    if psutil.pid_exists(process_id):
        for pid in pids:
            if pid == process_id:
                process_name = psutil.Process(pid).name()
    return process_name





# judge pid exsit
def pid_exsit(pid):
    return psutil.pid_exists(pid)



# judge process id exsit
def process_id_exsit(process_id):
    my_process_id_exsit=False
    pids=psutil.pids()
    for pid in pids:
        if pid == process_id:
            my_process_id_exsit = True
            break
    return my_process_id_exsit




# judge process name exsit
def process_name_exsit(process_name):
    my_process_name_exsit=False
    pids=psutil.pids()
    for pid in pids:
        if psutil.pid_exists(pid):
            if psutil.Process(pid).name().lower() == process_name.lower():
                my_process_name_exsit=True
                break
    return my_process_name_exsit






# judge process name exsit
def process_exsit(process_name):
    my_process_exsit=False
    pids=psutil.pids()
    for pid in pids:
        if psutil.pid_exists(pid):
            if psutil.Process(pid).name().lower()==process_name.lower():
                my_process_exsit=True
                break
    return my_process_exsit





# kill process by pid
def kill_process_with_id(process_id):
    try:
        pid_list = psutil.pids()
        for pid in pid_list:
            if pid==process_id:
                each_process = psutil.Process(process_id)
                each_process.terminate()
                each_process.wait(timeout=3)
                break
    except Exception as err:
        pass






# kill process by process name
# it supports cross platform linux and widnows
def kill_process_with_name(process_name):
    pid_list = psutil.pids()
    for pid in pid_list:
        try:
            each_pro = psutil.Process(pid)
            if process_name.lower() in each_pro.name().lower():
                each_pro.terminate()
                each_pro.wait(timeout=3)
        except Exception as err:
            pass





def kill_process(proc):
    """
    Kill a process and its children processes
    :param proc: Process class defined in psutil
    :return: None
    """
    try:
        children = proc.children()
        for child in children:
            try:
                child.terminate()
            except:
                pass
        gone, still_alive = psutil.wait_procs(children, timeout=3)
        for p in still_alive:
            p.kill()
        proc.kill()
    except:
        pass