import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def find_cubes(image,
               lower_hsv_threshold=np.array([170, 50, 70]),
               upper_hsv_threshold=np.array([190, 255, 255]),
               contour_color=(255, 0, 0)):
    _img_with_contours = image.copy()
    _blur_img = cv2.GaussianBlur(image, (99, 99), 0)
    _hsv_image = cv2.cvtColor(_blur_img, cv2.COLOR_BGR2HSV)
    _current_mask = cv2.inRange(_hsv_image, lower_hsv_threshold, upper_hsv_threshold)
    _contours, _hierarchy = cv2.findContours(_current_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _c in _contours:
        _M = cv2.moments(_c)
        _area = cv2.contourArea(_c)
        try:
            _cX = int((_M["m10"] / _M["m00"]))
            _cY = int((_M["m01"] / _M["m00"]))
        except:
            _cX = 0
            _cY = 0

        if _area >= 30000:
            print("area:", _area)

            cv2.drawContours(_img_with_contours, [_c], -1, (0, 255, 0), 11)
            cv2.putText(_img_with_contours, "cube", (_cX, _cY), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 11)
    return _img_with_contours


def find_cones(image,
               lower_hsv_threshold=np.array([70, 150, 50]),
               upper_hsv_threshold=np.array([110, 255, 255]),
               contour_color=(255, 0, 0)):
    _img_with_contours = image.copy()
    _blur_img = cv2.GaussianBlur(image, (199, 199), 0)
    _hsv_image = cv2.cvtColor(_blur_img, cv2.COLOR_BGR2HSV)
    _current_mask = cv2.inRange(_hsv_image, lower_hsv_threshold, upper_hsv_threshold)
    _contours, _hierarchy = cv2.findContours(_current_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for _c in _contours:
        _M = cv2.moments(_c)
        _area = cv2.contourArea(_c)
        try:
            _cX = int((_M["m10"] / _M["m00"]))
            _cY = int((_M["m01"] / _M["m00"]))
        except:
            _cX = 0
            _cY = 0

        if _area >= 30000:
            print("area:", _area)

            cv2.drawContours(_img_with_contours, [_c], -1, (0, 255, 0), 11)
            cv2.putText(_img_with_contours, "cone", (_cX, _cY), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 11)
    return _img_with_contours


img = mpimg.imread('test_image13.jpg')
new_img = find_cubes(img)
plt.figure(figsize=(6, 6))
plt.imshow(new_img, cmap='gray')
plt.show()

