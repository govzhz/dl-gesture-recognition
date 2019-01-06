import os
import re
import cv2
import functools
import subprocess
import moviepy.editor as mpy

import torch.nn.parallel
import torch.optim
from net.models import TSN
from ops.transforms import *
from torch.nn import functional as F

# options
# parser = argparse.ArgumentParser(description="test TRN on a single video")
# parser.add_argument('--video_file', type=str, default=None)
# parser.add_argument('--modality', type=str, default='RGB',
#                     choices=['RGB', 'Flow', 'RGBDiff'], )
# parser.add_argument('--dataset', type=str, default='jester',
#                     choices=['something', 'jester', 'moments'])
# parser.add_argument('--rendered_output', type=str, default=None)
# parser.add_argument('--arch', type=str, default="BNInception")
# parser.add_argument('--input_size', type=int, default=224)
# parser.add_argument('--test_segments', type=int, default=8)
# parser.add_argument('--img_feature_dim', type=int, default=256)
# parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
# parser.add_argument('--weight', type=str, default="pretrain/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar")
#
# args = parser.parse_args()

video_file = None
modality = "RGB"
dataset = "jester"
rendered_output = None
arch = "BNInception"
input_size = 224
test_segments = 8
img_feature_dim = 256
consensus_type = "TRNmultiscale"
weight = "model/TRN_jester_RGB_BNInception_TRNmultiscale_segment8_best.pth.tar"


def singleton(cls):
    ''' Use class as singleton. '''

    cls.__new_original__ = cls.__new__

    @functools.wraps(cls.__new__)
    def singleton_new(cls, *args, **kw):
        it =  cls.__dict__.get('__it__')
        if it is not None:
            return it

        cls.__it__ = it = cls.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__

    return cls



class Predict(object):

    def __init__(self):
        # Get dataset categories.
        categories_file = 'model/{}_categories.txt'.format(dataset)
        self.categories = [line.rstrip() for line in open(categories_file, 'r').readlines()]
        num_class = len(self.categories)

        # Load model.
        self.net = TSN(num_class,
                  test_segments,
                  modality,
                  base_model=arch,
                  consensus_type=consensus_type,
                  img_feature_dim=img_feature_dim, print_spec=False)

        weights = weight
        checkpoint = torch.load(weights)
        # print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint['state_dict'].items())}
        self.net.load_state_dict(base_dict)
        self.net.cuda().eval()

        # Initialize frame transforms.

        self.transform = torchvision.transforms.Compose([
            GroupOverSample(self.net.input_size, self.net.scale_size),
            Stack(roll=(arch in ['BNInception', 'InceptionV3'])),
            ToTorchFormatTensor(div=(arch not in ['BNInception', 'InceptionV3'])),
            GroupNormalize(self.net.input_mean, self.net.input_std),
        ])

        print("done")

    def runWithDiskImage(self, frame_folder):
        """从硬盘中的文件读取图片帧"""
        # Obtain video frames
        if frame_folder is not None:
            print('Loading frames in %s' % frame_folder)
            import glob
            # here make sure after sorting the frame paths have the correct temporal order
            frame_paths = sorted(glob.glob(os.path.join(frame_folder, '*.jpg')))
            frames = load_frames(frame_paths)
        else:
            print('Extracting frames using ffmpeg...')
            frames = extract_frames(video_file, test_segments)
        return self.runWithMemoryImage(frames)

    def runWithMemoryImage(self, frames):
        """"从内存中获取图片帧
        
        frames: 列表，每个元素为PIL对象或其子类
        """
        # Make video prediction.
        data = self.transform(frames)
        input_var = torch.autograd.Variable(data.view(-1, 3, data.size(1), data.size(2)),
                                            volatile=True).unsqueeze(0).cuda()
        logits = self.net(input_var)
        h_x = torch.mean(F.softmax(logits, 1), dim=0).data
        probs, idx = h_x.sort(0, True)

        # Output the prediction.
        res = dict()
        for i in range(0, 5):
            res[self.categories[idx[i]]] = "%.3f" % probs[i]
            print('{:.3f} -> {}'.format(probs[i], self.categories[idx[i]]))

        # Render output frames with prediction text.
        if rendered_output is not None:
            prediction = self.categories[idx[0]]
            rendered_frames = render_frames(frames, prediction)
            clip = mpy.ImageSequenceClip(rendered_frames, fps=4)
            clip.write_videofile(rendered_output)

        return res


def extract_frames(video_file, num_frames=8):
    try:
        os.makedirs(os.path.join(os.getcwd(), 'frames'))
    except OSError:
        pass

    output = subprocess.Popen(['ffmpeg', '-i', video_file],
                              stderr=subprocess.PIPE).communicate()
    # Search and parse 'Duration: 00:05:24.13,' from ffmpeg stderr.
    re_duration = re.compile('Duration: (.*?)\.')
    duration = re_duration.search(str(output[1])).groups()[0]

    seconds = functools.reduce(lambda x, y: x * 60 + y,
                               map(int, duration.split(':')))
    rate = num_frames / float(seconds)

    output = subprocess.Popen(['ffmpeg', '-i', video_file,
                               '-vf', 'fps={}'.format(rate),
                               '-vframes', str(num_frames),
                               '-loglevel', 'panic',
                               'frames/%d.jpg']).communicate()
    frame_paths = sorted([os.path.join('frames', frame)
                          for frame in os.listdir('frames')])

    frames = load_frames(frame_paths)
    subprocess.call(['rm', '-rf', 'frames'])
    return frames


def load_frames(frame_paths, num_frames=8):
    in_frame_cnt = len(frame_paths)

    if in_frame_cnt >= num_frames:
        seleted_frames = np.zeros(num_frames)
        scale = (in_frame_cnt - 1) * 1.0 / (num_frames - 1)
        if int(math.floor(scale)) == 0:
            seleted_frames[:in_frame_cnt] = np.arange(0, in_frame_cnt)
            seleted_frames[in_frame_cnt:] = in_frame_cnt - 1
        else:
            seleted_frames[::] = np.floor(scale * np.arange(0, num_frames))

        frames = list()
        for index in seleted_frames.astype(int):
            frames.append(Image.open(frame_paths[index]).convert('RGB'))
    else:
        raise ValueError('Video must have at least {} frames'.format(num_frames))

    return frames


def render_frames(frames, prediction):
    rendered_frames = []
    for frame in frames:
        img = np.array(frame)
        height, width, _ = img.shape
        cv2.putText(img, prediction,
                    (1, int(height / 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)
        rendered_frames.append(img)
    return rendered_frames


if __name__ == "__main__":
    predict = Predict()
    predict.runWithDiskImage("/home/zz/workspace/GestureRec/frames")