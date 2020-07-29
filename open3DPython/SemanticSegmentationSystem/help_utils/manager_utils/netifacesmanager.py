#!/usr/bin/python
# -*- coding: UTF-8 -*-




import netifaces
import subprocess as sp





# if returncode is 0 net is successful
def ping_baidu():
    ping_baidu_command_list = ["ping","-c","5","-w","5000","www.baidu.com"]
    try:
        ping_baidu_command_returncode=subprocess.check_call(ping_baidu_command_list)
        return ping_baidu_command_returncode
    except subprocess.CalledProcessError as sperror:
        return sperror.returncode



# if returncode is 0 net is successful
def ping_163():
    ping_163_command_list = ["ping","-c","5","-w","5000","www.163.com"]
    try:
        ping_163_command_returncode=subprocess.check_call(ping_163_command_list)
        return ping_163_command_returncode
    except subprocess.CalledProcessError as sperror:
        return sperror.returncode



# if returncode is 0 net is successful
def ping_taobao():
    ping_taobao_command_list = ["ping","-c","5","-w","5000","www.taobao.com"]
    try:
        ping_taobao_command_returncode=subprocess.check_call(ping_taobao_command_list)
        return ping_taobao_command_returncode
    except subprocess.CalledProcessError as sperror:
        return sperror.returncode






# linux ethernet name:eth0
# we can get it by ifconfig command
# we can get it by netifaces.interfaces()
# if we do not connect lan we will not have eth0 ip address when reboot
# and ifconfig eth0 will not have RUNNING
# and netifaces.ifaddresses("eth0") dict will not have 2
# netifaces.AF_INET=2
# netifaces.AF_INET6=10
# netifaces.AF_LINK=17
def judge_ethernet_ipv4_exsits(ethernet_name):
    ethernet_exsits=False
    net_interfaces_info_dict=netifaces.ifaddresses(ethernet_name)
    for key in net_interfaces_info_dict.keys():
        if key==netifaces.AF_INET:
            ethernet_exsits = True
            break
    return ethernet_exsits




# linux ethernet name:eth0
# we can get it by ifconfig command
# we can get it by netifaces.interfaces()
# if we do not connect lan we will not have eth0 ip address when reboot
# and ifconfig eth0 will not have RUNNING
# and netifaces.ifaddresses("eth0") dict will not have 2
# netifaces.AF_INET=2
# netifaces.AF_INET6=10
# netifaces.AF_LINK=17
def judge_ethernet_ipv6_exsits(ethernet_name):
    ethernet_exsits=False
    net_interfaces_info_dict=netifaces.ifaddresses(ethernet_name)
    for key in net_interfaces_info_dict.keys():
        if key==netifaces.AF_INET6:
            ethernet_exsits = True
            break
    return ethernet_exsits





# linux ethernet name:eth0
# we can get it by ifconfig command
# we can get it by netifaces.interfaces()
# if we do not connect lan we will not have eth0 ip address when reboot
# and ifconfig eth0 will not have RUNNING
# and netifaces.ifaddresses("eth0") dict will not have 2
# netifaces.AF_INET=2
# netifaces.AF_INET6=10
# netifaces.AF_LINK=17
# ifconfig_ethernet_output is <class 'bytes'> is not <class 'str'>
def judge_ethernet_is_running(ethernet_name):
    ifconfig_ethernet_list=["ifconfig",ethernet_name]
    ifconfig_ethernet_output_bytes=sp.check_output(ifconfig_ethernet_list,shell=False)
    ifconfig_ethernet_output_str=ifconfig_ethernet_output_bytes.decode("utf-8")
    if "RUNNING" in ifconfig_ethernet_output_str:
        return True
    else:
        return False










