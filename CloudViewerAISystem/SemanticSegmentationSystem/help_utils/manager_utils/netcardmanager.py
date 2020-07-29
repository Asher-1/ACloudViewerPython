#!/usr/bin/python
# -*- coding: UTF-8 -*-




from psutil import net_if_addrs




'''
get the wlan mac address
this can get all netcard information in your computer
for k, v in net_if_addrs().items():
    if "wlan" in str(k).lower():
        for item in v:
            print(item)
            print("netcard address :{0}".format(item[1]))
            print("====================================")
'''
def get_wlan_mac_address():
    wlan_mac_address = ''
    for k, v in net_if_addrs().items():
        if "wlan" in str(k).lower():
            for item in v:
                address = item[1]
                if '-' in address or ':' in address and len(address) == 17:
                    wlan_mac_address = address
    return wlan_mac_address
























