from flask import Flask, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from ops.predict import Predict
import json
import time
from PIL import Image
from io import BytesIO
import os

app = Flask(__name__)
predict = Predict()

photos = UploadSet('photos', IMAGES)

# 文件储存地址
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(os.getcwd(), "frames")
configure_uploads(app, photos)
# 文件大小限制，默认为16MB
patch_request_class(app)

"""API返回包装类"""
class Result(object):
    def __init__(self):
        self.code = 0
        self.msg = "success"
        self.data = None

    @staticmethod
    def success():
        return Result()

    @staticmethod
    def successReturnData(data):
        res = Result()
        res.data = data
        return res

    @staticmethod
    def failed(code, msg):
        res = Result()
        res.code = code
        res.msg = msg
        return res


@app.route('/upload', methods=['POST'])
def upload():
    """上传图片流，保存在本地。支持以下两种方式：
        1. 上传单张图片流
            img = Image.open("xxx.jpg")
            imgByteArr = io.BytesIO()
            img.save(imgByteArr, format="PNG")
            imgByteArr = imgByteArr.getvalue()
            requests.post("ip_address/upload", files={"image": ("xxx.jpg", imgByteArr)})
        2. 上传批量图片流
            imgs = list()
            for index in range(8):
                img = Image.open("frames/%06d.jpg" % index)
                imgByteArr = io.BytesIO()
                img.save(imgByteArr, format="PNG")
                imgByteArr = imgByteArr.getvalue()
                imgs.append(imgByteArr)
            requests.post("ip_address/upload", data=imgs)
            
    :return: json字符串
    """
    start_time = time.time()
    if request.method == 'POST':
        try:
            if 'image' in request.files:
                file = request.files['image']
                image = Image.open(BytesIO(file.read()))
                image.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], file.filename))
            else:
                data = request.get_data()
                imgs = data.split(b'\ngesture_train')
                for index, img in enumerate(imgs):
                    Image.open(BytesIO(img)).save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], "%06d.jpg" % index))
        except Exception as e:
            print(e)
            return json.dumps(Result.failed("-1", "upload failed").__dict__)
        print("upload time: %.4f" % (time.time() - start_time))
        return json.dumps(Result.success().__dict__)
    else:
        return json.dumps(Result.failed("-1", "method failed").__dict__)


@app.route("/remove", methods=['GET'])
def remove():
    """客户端每次准备上传图片前需要清空服务器数据"""
    try:
        for filename in os.listdir(app.config['UPLOADED_PHOTOS_DEST']):
            path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
            os.remove(path)
    except Exception as e:
        print(e)
        return json.dumps(Result.failed("-1", "remove failed").__dict__)
    return json.dumps(Result.success().__dict__)


@app.route("/category_network", methods=['GET'])
def category_network():
    """对上传图片进行分类，在调用该方法前请确保已正确上传所有图片
        因为网络原因，远程服务传的图片为实时传输，因此最终需要服务器进行筛选
    @see server.upload
    :return: json字符串
    """
    start_time = time.time()
    if request.method == 'GET':
        try:
            res = json.dumps(predict.runWithDiskImage(app.config['UPLOADED_PHOTOS_DEST']))
            print(res)
        except Exception as e:
            print(e)
            return json.dumps(Result.failed("-1", "get gesture category failed").__dict__)
        print("category time: %.4f" % (time.time() - start_time))
        return json.dumps(Result.successReturnData(res).__dict__)
    else:
        return json.dumps(Result.failed("-1", "method failed").__dict__)


@app.route("/category_local", methods=['GET'])
def category_local():
    """对上传图片进行分类，在调用该方法前请确保已正确上传所有图片
        本地识别客户端会筛选好指定图片，不需要服务器处理
    @see server.upload
    :return: json字符串
    """
    start_time = time.time()
    if request.method == 'GET':
        try:
            frames = [Image.open(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], "%06d.jpg" % index)) for index in range(8)]
            res = json.dumps(predict.runWithMemoryImage(frames))
        except Exception as e:
            print(e)
            return json.dumps(Result.failed("-1", "get gesture category failed").__dict__)
        print("category time: %.4f" % (time.time() - start_time))
        return json.dumps(Result.successReturnData(res).__dict__)
    else:
        return json.dumps(Result.failed("-1", "method failed").__dict__)


@app.route("/old", methods=['POST'])
def run_old():
    """客户端一次性上传8张图片，服务器识别分类返回（在网络环境下延迟过高，仅适合本地测试）"""
    start_time = time.time()
    if request.method == 'POST':
        data = request.get_data()
        imgs = data.split(b'\ngesture_train')
        frames = [Image.open(BytesIO(img)) for img in imgs]
        res = json.dumps(predict.runWithMemoryImage(frames))
        print("time: ", time.time() - start_time)
        return res
    else:
        return 'Error'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True)