#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
@Author: zz
@Date  : 2018/3/18
@Desc  : 
    固定队列
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class LimitQueue(object):

    def __init__(self, limit):
        self._queue = queue.Queue(maxsize=limit)
        self._lock = threading.Lock()
        self._aliveSum = 0
        self._limit = limit

    def put(self, item):
        self._lock.acquire()

        if self._queue.qsize() >= self._limit:
            remove_item = self._queue.get()
            self._aliveSum = max(0, self._aliveSum + (item - remove_item))
        else:
            self._aliveSum += item
        self._queue.put(item)

        self._lock.release()

    def getAliveSum(self):
        """返回当前队列中的总值（非负）"""
        return self._aliveSum


if __name__ == "__main__":

    def putToLimitQueue(limitQueue, item):
        limitQueue.put(item)
        print(limitQueue.getAliveSum())

    import random
    limitQueue = LimitQueue(5)
    threadPoolExecutor = ThreadPoolExecutor(max_workers=8)
    while True:
        threadPoolExecutor.submit(putToLimitQueue, limitQueue, random.randint(0, 1))