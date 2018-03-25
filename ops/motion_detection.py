#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2

class DiffMotionDetector:
    """Motion is detected through the difference between 
       the background (static) and the foregroung (dynamic).

    This class calculated the absolute difference between two frames.
    The first one is a static frame which represent the background 
    and the second is the image containing the moving object.
    The resulting mask is passed to a threshold and cleaned from noise. 
    """

    def __init__(self):
        """Init the color detector object.

    """
        self.background_gray = None

    def setBackground(self, frame):
        """Set the BGR image used as template during the pixel selection

        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify. the template
        is internally stored as an HSV image.
        @param frame the template to use in the algorithm
        """
        if(frame is None): return None 
        self.background_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def getBackground(self):
        """Get the BGR image used as template during the pixel selection

        The template can be a spedific region of interest of the main
        frame or a representative color scheme to identify.
        """
        if(self.background_gray is None): 
            return None
        else:
            return cv2.cvtColor(self.background_gray, cv2.COLOR_GRAY2BGR)

    def returnMask(self, foreground_image, threshold=25):
        """Return the binary image after the detection process

        @param foreground_image the frame to check
        @param threshold the value used for filtering the pixels after the absdiff
        """
        if(foreground_image is None):
            return None
        foreground_gray = cv2.cvtColor(foreground_image, cv2.COLOR_BGR2GRAY)
        delta_image = cv2.absdiff(self.background_gray, foreground_gray)
        threshold_image = cv2.threshold(delta_image, threshold, 255, cv2.THRESH_BINARY)[1]
        return threshold_image


class MogMotionDetector:
    """Motion is detected through the Mixtures of Gaussian (MOG) 

    This class is the implementation of the article "An Improved 
    Adaptive Background Mixture Model for Realtime Tracking with 
    Shadow Detection" by  KaewTraKulPong and Bowden (2008).

    ABSTRACT: Real-time segmentation of moving regions in image 
    sequences is a fundamental step in many vision systems 
    including automated visual surveillance, human-machine 
    interface, and very low-bandwidth telecommunications. A 
    typical method is background subtraction. Many background 
    models have been introduced to deal with different problems. 
    One of the successful solutions to these problems is to use a
    multi-colour background model per pixel proposed by Grimson 
    et al [1,2,3]. However, the method suffers from slow learning
    at the beginning, especially in busy environments. In addition,
    it can not distinguish between moving shadows and moving objects. 
    This paper presents a method which improves this adaptive 
    background mixture model. By reinvestigating the update equations,
    we utilise different equations at different phases. This allows
    our system learn faster and more accurately as well as adapt 
    effectively to changing environments. A shadow detection scheme
    is also introduced in this paper. It is based on a computational 
    colour space that makes use of our background model. A comparison
    has been made between the two algorithms. The results show the 
    speed of learning and the accuracy of the model using our update 
    algorithm over the Grimson et al tracker. When incorporate with 
    the shadow detection, our method results in far better segmentation
    than that of Grimson et al.
    """

    def __init__(self, history=10, numberMixtures=3, backgroundRatio=0.6, noise=20):
        """Init the color detector object.

        @param history lenght of the history
        @param numberMixtures The maximum number of Gaussian Mixture components allowed.
            Each pixel in the scene is modelled by a mixture of K Gaussian distributions.
            This value should be a small number from 3 to 5.
        @param backgroundRation define a threshold which specifies if a component has to be included
            into the foreground or not. It is the minimum fraction of the background model. 
            In other words, it is the minimum prior probability that the background is in the scene.
        @param noise specifies the noise strenght
        """
        self.BackgroundSubtractorMOG = cv2.bgsegm.createBackgroundSubtractorMOG(history, numberMixtures, backgroundRatio, noise)


    def returnMask(self, foreground_image):
        """Return the binary image after the detection process

        @param foreground_image the frame to check
        @param threshold the value used for filtering the pixels after the absdiff
        """
        return self.BackgroundSubtractorMOG.apply(foreground_image)

class Mog2MotionDetector:
    """Motion is detected through the Imporved Mixtures of Gaussian (MOG) 

    This class is the implementation of the article "Improved Adaptive 
    Gaussian Mixture Model for Background Subtraction" by Zoran Zivkovic.

    ABSTRACT: Background subtraction is a common computer vision task. 
    We analyze the usual pixel-level approach. We develop an efficient
    adaptive algorithm using Gaussian mixture probability density. 
    Recursive equations are used to constantly update the parameters
    and but also to simultaneously select the appropriate number of 
    components for each pixel.
    """

    def __init__(self):
        """Init the color detector object.

        """
        self.BackgroundSubtractorMOG2 = cv2.bgsegm.createBackgroundSubtractorMOG()


    def returnMask(self, foreground_image):
        """Return the binary image after the detection process

        @param foreground_image the frame to check
        """
        #Since the MOG2 returns shadows with value 127 we have to
        #filter these values in order to have a binary mask
        img = self.BackgroundSubtractorMOG2.apply(foreground_image)
        ret, thresh = cv2.threshold(img, 126, 255,cv2.THRESH_BINARY)
        return thresh 

    def returnGreyscaleMask(self, foreground_image):
        """Return the greyscale image after the detection process

        The MOG2 can return shadows. The pixels associated with
        shadows have value 127. This mask is not a classic binary
        mask since it incorporates the shadow pixels.
        @param foreground_image the frame to check
        """
        return self.BackgroundSubtractorMOG2.apply(foreground_image)


class KNNMotionDetector:

    def __init__(self):
        """Init the color detector object.

        """
        self.BackgroundSubtractorKNN = cv2.createBackgroundSubtractorKNN(detectShadows=True)


    def returnMask(self, foreground_image):
        """Return the binary image after the detection process

        @param foreground_image the frame to check
        """
        #Since the MOG2 returns shadows with value 127 we have to
        #filter these values in order to have a binary mask
        img = self.BackgroundSubtractorKNN.apply(foreground_image)
        ret, thresh = cv2.threshold(img, 244, 255,cv2.THRESH_BINARY)
        return thresh

    def returnGreyscaleMask(self, foreground_image):
        """Return the greyscale image after the detection process

        The MOG2 can return shadows. The pixels associated with
        shadows have value 127. This mask is not a classic binary
        mask since it incorporates the shadow pixels.
        @param foreground_image the frame to check
        """
        return self.BackgroundSubtractorKNN.apply(foreground_image)
