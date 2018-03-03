#!/usr/bin/env python
# encoding: utf-8
"""

@author:nikan

@file: request_sample.py

@time: 02/03/2018 11:27 PM
"""
import requests
from PIL import Image
import io

def test_hello():
    r = requests.get('http://0.0.0.0:5000/')
    assert r.text == 'Hello World!'

def test_gesture_train():
    route_name = 'gesture_train'
    allimgs = []
    for i in range(8):
        pic = '1.jpg'
        im = Image.open(pic)
        imgByteArr = io.BytesIO()
        im.save(imgByteArr, format='PNG')
        imgByteArr = imgByteArr.getvalue()
        allimgs.append(imgByteArr)
    allimg_bytes = b'\ngesture_train'.join(allimgs)
    r = requests.post(
        'http://0.0.0.0:5000/gesture_train',
        data= allimg_bytes
    )
    print(r.content)
    assert r.content == allimg_bytes
test_gesture_train()