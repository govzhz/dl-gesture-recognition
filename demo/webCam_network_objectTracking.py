#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import io
import json
import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from darkflow.net.build import TFNet

from ops.limitQueue import LimitQueue

"""
自动手势识别客户端Demo，支持27分类
    交互方式：
        当手出现在图像中时，动作开始；当手移出图像，动作结束
    特点：基于YOLO-v2的目标跟踪
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
    "Zooming Out With Two Fingers": "两个手指缩小"
}

parser = argparse.ArgumentParser(description='WebCam Network(Automatic)')
parser.add_argument('-s', '--server-address', default='http://127.0.0.1:5000', type=str,
                    help='手势识别服务器地址')
parser.add_argument('-d', '--device', default=0, type=int,
                    help='摄像头设备号')


# 图片上传大小（约27k）
UPLOAD_SIZE = (176, 100)
# 并发上传图片的线程池数
POOL_SIZE = 5
# 自动识别判定总帧数（即通过n帧来判定）
DETECT_FRAMES_TOTAL = 5
# 动作开始所需帧数（不大于总帧数）
DETECT_FRAMES_START_ACTION = 4
# 动作结束所需帧数（不大于总帧数）
DETECT_FRAMES_END_ACTION = 4
# darkflow所需配置
DARKFLOW_OPTIONS = {
    "model": "cfg/yolo-gestures.cfg",
    "load": "model/yolo-gestures_15000.weights",
    "threshold": 0.1,
    'gpu': 0.5,
    "json": True
}


class GestureRec(object):
    def __init__(self, server_address, upload_size, device_index, pool_size, detect_frames_total,
                 detect_frames_start_action, detect_frames_end_action, darkflow_options, quit_key='q'):
        self._quit = quit_key  # 退出摄像头
        self._server_address = server_address  # 手势识别服务器地址
        self._upload_size = upload_size  # 上传图片大小
        self._detect_frames_total = detect_frames_total  # 自动识别判定总帧数
        self._detect_frames_start_action = detect_frames_start_action  # 动作开始所需帧数
        self._detect_frames_end_action = detect_frames_end_action  # 动作结束所需帧数

        self._save = None  # 是否开始保存帧画面
        self._label = None  # 分类预测标签
        self._pro = None  # 预测准确率
        self._new = True  # 第一次打开系统标志

        print("系统正在初始化...")
        self._tfnet = TFNet(darkflow_options)
        self._cap_video = cv2.VideoCapture(device_index)  # 摄像头句柄
        self._queue_draw = queue.Queue()  # 存放显示在界面的标签
        self._queue_detect = LimitQueue(self._detect_frames_total)  # 保存帧画面检测结果
        self._save_distance = self.get_save_distance()  # 保存帧画面间距
        self._pool = ThreadPoolExecutor(max_workers=pool_size)  # 线程池

    def get_save_distance(self):
        """计算保存帧画面间距"""
        # Number of frames to capture
        num_frames = 120

        print("Capturing {0} frames".format(num_frames))

        # Start time
        start = time.time()

        # Grab a few frames
        for i in range(0, num_frames):
            ret, frame = self._cap_video.read()
            # cv2.imwrite(r"E:\df\%06d.jpg" % i, frame)

        # End time
        end = time.time()

        # Time elapsed
        seconds = end - start
        print("Time taken : {0} seconds".format(seconds))

        # Calculate frames per second
        fps = num_frames / seconds
        distance = math.ceil(fps / 5)
        print("Estimated frames per second : {0}".format(fps))
        print("Save distance: %s" % distance)
        return distance

    # 捕捉视频
    def run(self):
        cnt_frames = 0  # 统计所有帧数
        cnt_upload_frames = 0  # 统计上传帧数
        while self._cap_video.isOpened():
            # 获取帧画面, 如果摄像头开启成功
            ret, frame = self._cap_video.read()
            cnt_frames += 1

            # 第一次加载系统提示
            if self._new:
                frame = self._draw(frame, "系统准备完毕")

            if ret:
                # 子线程检测是否存在手，将检测结果插入队列用于系统确定动作的开始和结束
                if (cnt_frames + 1) % self._save_distance == 0:
                    self._pool.submit(self._detect_frame, frame)

                # 如果系统判定动作开始，则开启子线程上传图片到服务器，直到动作结束
                if self._save and (cnt_frames + 1) % self._save_distance == 0:
                    self._pool.submit(self._upload_frame, cnt_upload_frames, frame)
                    cnt_upload_frames += 1

                # 主线程读取分类预测结果
                if not self._queue_draw.empty():
                    try:
                        self._label, self._pro = self._queue_draw.get_nowait()
                    except Exception as e:
                        print(e)

                # 显示预测结果
                if self._label is not None:
                    if self._pro == -1:
                        frame = self._draw(frame, self._label)
                    else:
                        frame = self._draw(frame, labels[self._label])
                    # cv2.putText(frame, labels[self._label] + " - " + str(self._pro),
                    #             (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

                # 显示图像
                cv2.imshow('Main', frame)

            # 按q退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            # 系统判定动作开始
            if self._save is None and self._queue_detect.getAliveSum() >= self._detect_frames_start_action:
                self._queue_draw.put(("请完成动作", -1))
                # 初始化
                self._label = None
                self._pro = None
                self._save = True  # 标识识别开始
                self._new = False
                # 清空服务器数据
                threading.Thread(target=self._remove).start()

            # 动作开始后，如果5帧以上识别手识别，动作结束
            if self._save and self._queue_detect.getAliveSum() <= \
                    (self._detect_frames_total - self._detect_frames_end_action):
                self._queue_draw.put(("正在分类中", -1))
                self._save = False  # 标识识别结束
                # 预测分类
                threading.Thread(target=self._predict).start()

        # 停止捕获视频
        self._cap_video.release()
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

    def _draw(self, frame, text):
        pil_im = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype("font/simhei.ttf", 30, encoding='utf-8')
        draw.text((30, 30), text, (0, 0, 255), font=font)
        frame = np.array(pil_im)
        return frame

    def _predict(self):
        """预测分类"""
        start_time = time.time()
        res = json.loads(requests.get(self._server_address + "/category_network").text)
        if res["code"] == 0:
            print("Category ok...[%.4f]" % (time.time() - start_time))
            print(res["data"])
            res = json.loads(res["data"])
            category = max(res, key=res.get)
            self._queue_draw.put((category, res[category]))
        else:
            print("Category failed...[%.4f]" % (time.time() - start_time))
            self._queue_draw.put(("动作过快，识别错误", -1))

        # 标识下一轮识别开始
        self._save = None

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


if __name__ == "__main__":
    args = parser.parse_args()
    gestureRec = GestureRec(server_address=args.server_address, upload_size=UPLOAD_SIZE, device_index=args.device,
               pool_size=POOL_SIZE, detect_frames_total=DETECT_FRAMES_TOTAL,
               detect_frames_start_action=DETECT_FRAMES_START_ACTION,
               detect_frames_end_action=DETECT_FRAMES_END_ACTION, darkflow_options=DARKFLOW_OPTIONS)
    gestureRec.run()
