#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2020/12/7 19:17
# @Author  : QXTD-LXH
# @Desc    :

from __future__ import print_function

import sys
sys.path.append("../")
import socket
import importlib
from _thread import start_new_thread
import requests
import time
import traceback
import json

importlib.reload(sys)

SERVER_IP = "0.0.0.0"
SERVER_PORT = 80
TARGET_RUL = "http://apple55bc.picp.io:2112/qianyan"

print("starting conversation server ...")
print("binding socket ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096 * 20)
s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096 * 20)
bufsize = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print( "Buffer size [After]: %d" %bufsize)
#Bind socket to local host and port
try:
    s.bind((SERVER_IP, SERVER_PORT))
except socket.error as msg:
    print("Bind failed. Error Code : " + str(msg[0]) + " Message " + msg[1])
    exit()
#Start listening on socket
s.listen(10)
print("bind socket success !")

print("start conversation server success !")


def clientthread(conn, addr):
    """
    client thread
    """
    logstr = "addr:" + addr[0]+ "_" + str(addr[1])
    b_time = time.time()
    try:
        #Receiving from client
        while True:
            time.sleep(0.1)
            param_ori = conn.recv(4096 * 20)
            if time.time() - b_time > 5.0:
                raise ValueError('Cant Decode: {}'.format(param_ori))
            try:
                json.loads(param_ori)
            except json.JSONDecodeError:
                print('Cant Decode: {}'.format(param_ori))
                continue
            param = param_ori.decode('utf-8', "ignore")
            print('Get param {}'.format(param))
            break
        if param is not None:
            response = requests.post(
                TARGET_RUL,
                params={'input': param}
            ).text
            logstr += "\tresponse:" + response
            conn.sendall(response.encode())
            # response = {
            #     "error code": 0,
            #     "response": response,
            # }
            # conn.sendall(json.dumps(response, ensure_ascii=False).encode())
        conn.close()
        print(logstr + "\n")
    except Exception as e:
        exc_info = 'Cal Exception: {}'.format(e)
        print(logstr + "\n", e)
        print('==========')
        response = {
            "error code": 1,
            "response": "error: {}".format(exc_info)
        }
        conn.sendall(json.dumps(response, ensure_ascii=False).encode())
        conn.close()
        raise


while True:
    conn, addr = s.accept()
    start_new_thread(clientthread, (conn, addr))
s.close()
