#!/usr/bin/python
# -*- coding: UTF-8 -*-




import socket




'''
host_name : ip or domain
if the network is connected
socket.gethostbyname(host_name) can get ip address
return : 
if the network is connected 
it will return True
else it will return Flase
'''
def network_is_connected(host_name):
    try:
        ip_address = socket.gethostbyname(host_name)
        if ip_address:
            return True
    except Exception as err:
        return False











