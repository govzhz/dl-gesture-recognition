#!/usr/bin/python
# -*- coding: UTF-8 -*-

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import requests
import json
import math
import io
import threading
from threading import Condition, Thread
import queue
import time

"""
手势识别客户端Demo，支持27分类
    使用方式：第一次按下 s 键开始（左上角红字消失），第二次按下 s 键停止（然后等待识别结果出现在左上角），循环往复。
    交互方式：
              1. 客户端采用多线程上传图片，并等待服务器返回
              2. 服务器端分别对所有图片保存本地，返回响应
              3. 客户端得到所有图片上传完毕信息，接着向服务器请求分类结果
              4. 服务器返回分类结果
    特点：推荐该交互方式，可设置超时，并且速度象相对较快
"""

labels = {
    "Doing other things": "正在做其他事（非手势）",
    "Drumming Fingers": "打击手指",
    "No gesture": "无手势",
    "Pulling Hand In": "拉近手",
    "Pushing Hand Away": "推远手",
    "Pulling Two Fingers In": "拉近两个手指",
    "Pushing Two Fingers Away": "推远两个手指",
    "Rolling Hand Backward": "",
    "Rolling Hand Forward": "",
    "Shaking Hand": "抖动手",
    "Sliding Two Fingers Down": "向下滑动两个手指",
    "Sliding Two Fingers Left": "向左滑动两个手指",
    "Sliding Two Fingers Right": "向右滑动两个手指",
    "Sliding Two Fingers Up": "向上滑动两个手指",
    "Stop Sign": "停止信号",
    "Swiping Down": "向下滑动",
    "Swiping Left": "向左滑动",
    "Swiping Right": "向右滑动",
    "Swiping Up": "向上滑动",
    "Thumb Down": "拇指向下",
    "Thumb Up": "竖起大拇指",
    "Turning Hand Clockwise": "顺时针转动手",
    "Turning Hand Counterclockwise": "逆时针转动手",
    "Zooming In With Full Hand": "整只手放大",
    "Zooming In With Two Fingers": "两个手指放大",
    "Zooming Out With Full Hand": "整只手缩小",
    "Zooming Out With Two Fingers": "两个手指缩小",
    "图片上传失败": "图片上传失败",
    "未知分类错误": "未知分类错误"
}


class VideoCapture(object):

    def __init__(self, server_address, upload_size=(176, 100), device_index=0, quit_key='q'):
        self._device_index = device_index  # 设备索引号或者视频
        self._quit = quit_key  # 退出摄像头
        self._server_address = server_address  # 手势识别服务器地址
        self._upload_size = upload_size  # 上传图片大小
        self._save = False  # 是否开始保存帧画面
        self._label = None  # 预测标签
        self._pro = None  # 预测准确率
        self._queue = queue.Queue()  # 保存预测结果
        self._new = True  # 第一次打开系统标志
        self._frames = list()

    # 捕捉视频
    def run(self):
        cap = cv2.VideoCapture(self._device_index)

        while cap.isOpened():
            # 获取帧画面, 如果摄像头开启成功
            ret, frame = cap.read()

            # 第一次加载系统提示
            if self._new:
                frame = self._draw(frame, "系统准备完毕")

            # 对帧画面操作
            if ret:
                if self._save:
                    self._frames.append(frame)

                # 读取预测结果
                if not self._queue.empty():
                    try:
                        self._label, self._pro = self._queue.get_nowait()
                    except Exception as e:
                        print(e)

                # 显示预测结果
                if self._label is not None:
                    frame = self._draw(frame, labels[self._label])
                    # cv2.putText(frame, labels[self._label] + " - " + str(self._pro),
                    #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

                # 显示图像
                cv2.imshow('Main', frame)

            # 按q退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # 按s开始捕捉，再次按下结束捕捉同时开始预测
            if key == ord('s'):
                if not self._save:
                    # 初始化
                    self._frames = list()
                    self._label = None
                    self._pro = None
                    self._save = True
                    self._new = False
                else:
                    self._save = False
                    threading.Thread(target=self._predict).start()

        # 停止捕获视频
        cap.release()
        cv2.destroyAllWindows()

    def load_frames(self, inFrames, num_frames=8):
        in_frame_cnt = len(inFrames)

        if in_frame_cnt >= num_frames:
            seleted_frames = np.zeros(num_frames)
            scale = (in_frame_cnt - 1) * 1.0 / (num_frames - 1)
            if int(math.floor(scale)) == 0:
                seleted_frames[:in_frame_cnt] = np.arange(0, in_frame_cnt)
                seleted_frames[in_frame_cnt:] = in_frame_cnt - 1
            else:
                seleted_frames[::] = np.floor(scale * np.arange(0, num_frames))

            outFrames = [inFrames[index] for index in seleted_frames.astype(int)]
        else:
            raise ValueError('Video must have at least {} frames'.format(num_frames))

        return outFrames

    def _predict(self):
        start_time = time.time()
        # 上传图片
        frames = self.load_frames(self._frames)
        latch = CountDownLatch(len(frames))
        q = queue.Queue()

        for index, frame in enumerate(frames):
            UploadThread(index, frame, q, latch, self._server_address, self._upload_size).start()

        latch.await()
        uploadOK = True
        while not q.empty():
            res = json.loads(q.get())
            if res["code"] != 0:
                uploadOK = False
                break

        if not uploadOK:
            print("Upload images failed...[%.4f]" % (time.time() - start_time))
            self._queue.put(("图片上传失败", 0))
        else:
            print("Upload images OK...[%.4f]" % (time.time() - start_time))

            # 分类
            start_time = time.time()
            res = json.loads(requests.get(self._server_address + "/category_local").text)
            if res["code"] == 0:
                print("Category ok...[%.4f]" % (time.time() - start_time))
                res = json.loads(res["data"])
                category = max(res, key=res.get)
                self._queue.put((category, res[category]))
            else:
                print("Category failed...[%.4f]" % (time.time() - start_time))
                self._queue.put(("未知分类错误", 0))

    def _draw(self, frame, text):
        pil_im = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("../font/simhei.ttf", 30, encoding='utf-8')
        draw.text((30, 30), text, (0, 0, 255), font=font)
        frame = np.array(pil_im)
        return frame


class UploadThread(Thread):
    def __init__(self, index, img, q, latch, server_address, upload_size):
        Thread.__init__(self)
        self.index = index  # 索引
        self.img = img  # opencv捕捉的画面帧
        self.q = q  # 队列
        self.latch = latch  # 同步类
        self.server_address = server_address  # 服务器地址
        self.upload_size = upload_size  # 上传图片大小

    def run(self):
        start_time = time.time()
        img = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        img = img.resize(self.upload_size, Image.ANTIALIAS)
        imgByteArr = io.BytesIO()
        img.save(imgByteArr, format="PNG")
        imgByteArr = imgByteArr.getvalue()
        res = '{"code": -1}'
        try:
            res = requests.post(self.server_address + "/upload",
                                        files={"image": ("%06d.jpg" % int(self.index), imgByteArr)}, timeout=(0.1, 1)).text
        except Exception as e:
            print(e)
        self.q.put(res)
        print("%s: %s[%s]" % (threading.current_thread().getName(), res, time.time() - start_time))
        self.latch.countDown()


class CountDownLatch:
    def __init__(self, count):
        self.count = count
        self.condition = Condition()

    def await(self):
        try:
            self.condition.acquire()
            while self.count > 0:
                self.condition.wait()
        finally:
            self.condition.release()

    def countDown(self):
        try:
            self.condition.acquire()
            self.count -= 1
            self.condition.notifyAll()
        finally:
            self.condition.release()


if __name__ == "__main__":
    cap = VideoCapture(server_address="http://127.0.0.1:5000")
    cap.run()
