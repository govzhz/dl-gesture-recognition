#!/usr/bin/env python
# encoding: utf-8
"""

@author:nikan

@file: app.py

@time: 02/03/2018 11:03 PM
"""

from flask import Flask, request
import io


app = Flask(__name__)


@app.route("/")
def hello():
    return "Hello World!"

@app.route("/gesture_train", methods=['POST'])
def train():
    """
    POST 的是图片二进制，需要对图片进行处理
    :return:
    """
    if request.method == 'POST':
        # TODO:
        data = request.get_data()
        imgs = data.split(b'\ngesture_train')
        for img in imgs:
            print(img)
        print('get_data')
        return data
    else:
        return 'Error'
    return ""


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)