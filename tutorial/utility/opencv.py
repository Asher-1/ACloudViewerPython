# CloudViewer: www.erow.cn
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Utility/opencv.py


def initialize_opencv():
    opencv_installed = True
    try:
        import cv2
    except ImportError:
        pass
        print("OpenCV is not detected. Using Identity as an initial")
        opencv_installed = False
    if opencv_installed:
        print("OpenCV is detected. Using ORB + 5pt algorithm")
    return opencv_installed
