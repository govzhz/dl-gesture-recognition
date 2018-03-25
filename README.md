This project uses front-end separation, and the client has the following three forms of implementation:

- Semi-automatic gesture recognition, which means that the user determines the segmentation of continuous gestures
- Dynamic gesture recognition based on frame difference method
- Dynamic gesture recognition based on object tracking

**Note**: No server code is currently provided.

## Dependencies

Python3.6, OpenCV3.4 + opencv_contrib

```
$ pip install pillow
$ pip install requests
```

## Usage

### Semi-automatic

- server address: gesture recognition server address

```
$ python run_manual.py -s [server-address]
```

### Frame difference

you can choose the Background Subtraction Methods

- method: knn or mog
- threshold: The sum of the length and width of the contour identified by the algorithm is greater than the threshold is considered to be the hand

```
$ python run_frameDifferent -s [server-address] --method [method] --threshold [threshold]
```

### Object tracking

you need to install tensorflow1.6-gpu extra and [darkflow](https://github.com/thtrieu/darkflow), You can download it from [here](https://drive.google.com/open?id=1khaq-aWudYW_b4GC7R_tyzWiWL3AzJE9)

```
$ pip install tensorflow-gpu
$ pip install Cython
$ cd darkflow
$ pip install .

# Check whether the installation is complete
$ flow --h
```

and then download the [weight file](https://drive.google.com/open?id=1pcmIyYp1GcJOHNkzWPrPcg1tkMeCewTx) and [configuration file](https://drive.google.com/open?id=1nfp0LO-quY2LxiQ4zEQRdEVp6q6BzLG9), and place them in the `model` folder and `cfg` folder respectively. Finally, run

```
$ python run_objectDetection.py -s [server-address]
```