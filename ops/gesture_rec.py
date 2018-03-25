#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
from PIL import Image
import requests
import json
import io
import threading
import time
import math

labels = {
    "Doing other things": "非手势动作",
    "Drumming Fingers": "抖动手指",
    "No gesture": "非手势动作",
    "Pulling Two Fingers In": "拉近两个手指",
    "Pushing Two Fingers Away": "推远两个手指",
    "Shaking Hand": "抖动手",
    "Sliding Two Fingers Down": "向下滑动两个手指",
    "Swiping Left": "向左滑动",
    "Thumb Down": "拇指向下",
    "Thumb Up": "竖起大拇指",
    "Zooming In With Full Hand": "整只手放大",
    "Zooming In With Two Fingers": "两个手指放大",
    "Zooming Out With Full Hand": "整只手缩小",
    "Zooming Out With Two Fingers": "两个手指缩小",
}


class GestureRec(object):

    def __init__(self, server_address, upload_size, frame_distance, queue_draw, pool, tfnet=None, queue_detect=None):
        self._server_address = server_address  # 手势识别服务器地址
        self._upload_size = upload_size  # 上传图片大小
        self._frame_distance = frame_distance  # 帧画面间距
        self._queue_draw = queue_draw  # 保存显示结果
        self._pool = pool  # 线程池
        self._tfnet= tfnet  # 手势检测
        self._queue_detect = queue_detect  # 保存帧画面检测结果

    def check_upload(self, frame_total, upload_total, frame, is_upload):
        """ 检测是否上传帧画面

        :param frame_total: 当前视频捕捉总帧数
        :param upload_total: 当前上传总帧数
        :param frame: 帧画面
        :param is_upload: 是否上传
        """
        if is_upload[0] and (frame_total + 1) % self._frame_distance == 0:
            self._pool.submit(self._upload_frame, upload_total, frame)
            return upload_total + 1
        return upload_total

    def detection(self, frame_total, frame):
        """ 检测手势

        :param frame_total: 当前视频捕捉总帧数
        :param frame: 帧画面
        """
        if (frame_total + 1) % self._frame_distance == 0:
            self._pool.submit(self._detect_frame, frame)

    def check_draw_text(self):
        """ 得到显示文本，包括系统提示和预测标签

        :return: 如果无显示文本，则返回None
        """
        if not self._queue_draw.empty():
            try:
                return self._queue_draw.get_nowait()
            except Exception as e:
                print(e)

        return None

    def start_action(self):
        """ 开始动作"""
        threading.Thread(target=self._remove).start()

    def end_action(self, is_upload=None):
        """ 结束动作"""
        threading.Thread(target=self._predict, args=(is_upload, )).start()

    def _detect_frame(self, frame):
        """检测图片

        :param frame: 帧画面
        :return: json字符串
        """
        frame = cv2.resize(frame, self._upload_size, interpolation=cv2.INTER_AREA)
        res = self._tfnet.return_predict(frame)
        found = False
        for region in res:
            if region["label"] == "hand":
                found = True
                break
        if found:
            print("Found: %s" % res)
            self._queue_detect.put(1)
        else:
            print("Not found: %s" % res)
            self._queue_detect.put(0)

    def _upload_frame(self, index, frame):
        """上传图片

        :param index: 图片索引
        :param frame: 帧画面
        :return: json字符串
        """
        start_time = time.time()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize(self._upload_size, Image.ANTIALIAS)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="PNG")
        imgByteArr = imgByteArr.getvalue()
        res = None
        try:
            res = requests.post(self._server_address + "/upload",
                                files={"image": ("%06d.jpg" % int(index), imgByteArr)}, timeout=(0.1, 0.8)).text
            print("Upload: %s[%.4f]" % (res, time.time() - start_time))
        except Exception as e:
            print(e)
        return res

    def _predict(self, is_upload=None):
        """预测分类"""
        start_time = time.time()
        res = json.loads(requests.get(self._server_address + "/category_network").text)
        if res["code"] == 0:
            print("Category ok...[%.4f]" % (time.time() - start_time))
            res = json.loads(res["data"])
            translate_res = dict()
            for category in res:
                if category in labels:
                    translate_res[labels[category]] = math.floor(float(res[category]) * 100)

            if len(translate_res) == 0:
                self._queue_draw.put({"未知分类": -1})
            else:
                self._queue_draw.put(translate_res)
        else:
            print("Category failed...[%.4f]" % (time.time() - start_time))
            self._queue_draw.put({"动作过快": -1})

        # 服务于自动切分系统的上传标志
        if is_upload is not None:
            is_upload[0] = None

    def _remove(self):
        """清空服务器图片数据

        :return: json字符串
        """
        start_time = time.time()
        res = json.loads(requests.get(self._server_address + "/remove").text)
        if res["code"] == 0:
            print("Remove ok...[%.4f]" % (time.time() - start_time))
        else:
            print("Remove failed...[%.4f]" % (time.time() - start_time))
