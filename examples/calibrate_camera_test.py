import numpy as np

import sys
#sys.path.insert(0, '/home/busta/git/opencv/Debug/lib')
import cv2
import glob
from matplotlib import pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

conres_in = 7
conres_in2 = 7


objp = np.zeros((conres_in*conres_in2,3), np.float32)
objp[:,:2] = np.mgrid[0:conres_in,0:conres_in2].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('../datao/*.JPG')
img = None

for fname in images:

    #if img != None:
    #    break

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if gray.shape[1] > 1024:
        dsize = (1024, int(gray.shape[0] / (img.shape[1] / float(1024))))
        gray = cv2.resize(gray, dsize)
        img = cv2.resize(img, dsize)
    #gray = cv2.bilateralFilter(gray, 5, 5, 5)
    #cv2.imshow("gray", gray)
    #cv2.waitKey(0)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (conres_in,conres_in2), None, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (conres_in,conres_in2), corners2,ret)
        plt.imshow(img)
        plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread(images[0])
if img.shape[1] > 1024:
    dsize = (1024, int(img.shape[0] / (img.shape[1] / float(1024))))
    img = cv2.resize(img, dsize)

h,  w = img.shape[:2]
grid_img = np.zeros( (h, w), np.uint8 )

for x in range(0, w, 20):
    cv2.line(grid_img, (x, 0), (x, h), (255, 0, 0) )
for y in range(0, h, 20):
    cv2.line(grid_img, (0, y), (w, y), (255, 0, 0) )



newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
dstg = cv2.undistort(grid_img, mtx, dist, None, newcameramtx)

x,y,w,h = roi
#dst = dst[y:y+h, x:x+w]
cv2.imshow('calibresult.png',dst)
cv2.imshow('calibresult.png2',dstg)
cv2.waitKey(0)

