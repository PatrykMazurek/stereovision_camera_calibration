import numpy as np
import cv2
import glob
import time
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7*6,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.

imagescale = 1

def CalibrateCamera(pathtofiles, debugimage = False, imagescale = 1):
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(pathtofiles)

    import random
    MAX_IMAGES = 64
    if (len(images) > MAX_IMAGES):
        print("Too many images to calibrate, using {0} randomly selected images"
                .format(MAX_IMAGES))
        images = random.sample(images, MAX_IMAGES)

    images = images[0:21]  
    h = 0
    w = 0
    for fname in sorted(images):
        img = cv2.imread(fname)
        
        newX,newY = img.shape[1]*imagescale, img.shape[0]*imagescale
        img = cv2.resize(img,(int(newX),int(newY)))
      
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            
            corners2 = cv2.cornerSubPix(gray,corners,(11,10),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
            if(debugimage):
                cv2.imshow(fname,img)
                cv2.waitKey(500)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return (ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints, w, h)


retL, leftCameraMatrix, leftDistortionCoefficients, _, _, leftObjectPoints, leftImagePoints, w, h = CalibrateCamera('D:\studia\Python\publikacja\stereo_vision\images\L640\*.jpg', False, imagescale)

retR, rightCameraMatrix, rightDistortionCoefficients, _, _, rightObjectPoints, rightImagePoints, w, h = CalibrateCamera('D:\studia\Python\publikacja\stereo_vision\images\R640\*.jpg', False, imagescale)
#print("*********************************************")
#print(leftObjectPoints)

#print("*********************************************")
#print(rightObjectPoints)
#print("*********************************************")
#print(leftImagePoints)
#print("*********************************************")
#print(rightImagePoints)
#print("*********************************************")

print('************************************')
print(retL)
print(retR)

print('************************************')

objectPoints = leftObjectPoints

print("Calibrating cameras together...")
#focal length 9 mm
multiplayer = 1
imageSize = tuple([w * multiplayer, 
                   h * multiplayer])

TERMINATION_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)

termination_criteria_extrinsics = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.00001)

print('****************************')
print(len(leftImagePoints))
print(len(rightImagePoints))

(retval, _, _, _, _, rotationMatrix, translationVector, _, _) = cv2.stereoCalibrate(
        objectPoints, leftImagePoints, rightImagePoints,
        leftCameraMatrix, leftDistortionCoefficients,
        rightCameraMatrix, rightDistortionCoefficients,
        imageSize, criteria=termination_criteria_extrinsics,
        flags=cv2.CALIB_FIX_INTRINSIC)

print(retval)
print(rotationMatrix)
print(translationVector)

OPTIMIZE_ALPHA = -1

print("Rectifying cameras...")

(leftRectification, rightRectification, leftProjection, rightProjection,
        dispartityToDepthMap, leftROI, rightROI) = cv2.stereoRectify(
                leftCameraMatrix, leftDistortionCoefficients,
                rightCameraMatrix, rightDistortionCoefficients,
                imageSize, rotationMatrix, translationVector,
                None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, OPTIMIZE_ALPHA)
print(leftROI)

print("Saving calibration...")
leftMapX, leftMapY = cv2.initUndistortRectifyMap(leftCameraMatrix, leftDistortionCoefficients, leftRectification, leftProjection, (w * multiplayer, h * multiplayer), 5)
rightMapX, rightMapY = cv2.initUndistortRectifyMap(rightCameraMatrix, rightDistortionCoefficients,  rightRectification, rightProjection, (w * multiplayer, h * multiplayer), 5)

print(leftCameraMatrix)
print(rightCameraMatrix)




np.savez_compressed('D:\studia\Python\publikacja\stereo_vision\calib.param', imageSize=[w,h],
        leftMapX=leftMapX, leftMapY=leftMapY,
        rightMapX=rightMapX, rightMapY=rightMapY,
        leftProjection=leftProjection, rightProjection=rightProjection,
        leftCameraMatrix=leftCameraMatrix,rightCameraMatrix=rightCameraMatrix,
        dispartityToDepthMap = dispartityToDepthMap,
        leftROI=leftROI, rightROI=rightROI)

leftFrame = cv2.imread('D:\\studia\\Python\\publikacja\\stereo_vision\\images\\stereo_test_l.jpg')
rightFrame = cv2.imread('D:\\studia\\Python\\publikacja\\stereo_vision\\images\\stereo_test_r.jpg')

newX,newY = leftFrame.shape[1]*imagescale, leftFrame.shape[0]*imagescale
leftFrame = cv2.resize(leftFrame,(int(newX),int(newY)))

rightFrame = cv2.resize(rightFrame,(int(newX),int(newY)))

cv2.imshow("al", leftFrame)
cv2.imshow("ar", rightFrame)

REMAP_INTERPOLATION = cv2.INTER_LINEAR

fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_WIDTH_2 = 640 / 2
CAMERA_HEIGHT_2 = 480 / 2

cropPercent = 1
def getCenterOLD(image, mm):
    centerX = int(mm[0, 2]*2.0)
    centerY = int(mm[1, 2]*2.0)
    print(centerX)
    print(centerY)
    return image[(centerY - int(CAMERA_HEIGHT_2 * cropPercent)):(centerY + int(CAMERA_HEIGHT_2 * cropPercent)),
           (centerX - int(CAMERA_WIDTH_2 * cropPercent)):(centerX + int(CAMERA_WIDTH_2 * cropPercent))]


def getCenter(image):
    centerY = int(image.shape[0] / 2) - 150
    centerX = int(image.shape[1] / 2)

    return image[(centerY - int(CAMERA_HEIGHT_2 * cropPercent)):(centerY + int(CAMERA_HEIGHT_2 * cropPercent)),
           (centerX - int(CAMERA_WIDTH_2 * cropPercent)):(centerX + int(CAMERA_WIDTH_2 * cropPercent))]


grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)

stereo = cv2.StereoBM_create(numDisparities=16, 
                             blockSize=15)

NumDisparities = 256
NumDisparitiesHalf = int(NumDisparities / 2)
# https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html
stereo.setMinDisparity(1)
stereo.setNumDisparities(32)
stereo.setBlockSize(229) 
stereo.setSpeckleRange(2) 
stereo.setSpeckleWindowSize(50)  

disparity = stereo.compute(grayLeft, grayRight)

DEPTH_VISUALIZATION_SCALE = 512

disp = (disparity + 0) / DEPTH_VISUALIZATION_SCALE
disp = np.array(disp * 255, dtype=np.uint8)
# disp = cv2.cvtColor(disparity / DEPTH_VISUALIZATION_SCALE, cv2.CV_8UC1)
# https://www.learnopencv.com/applycolormap-for-pseudocoloring-in-opencv-c-python/
im_color = cv2.applyColorMap(disp, cv2.COLORMAP_HOT)
cv2.imshow('depth', im_color)
fixedRight_copy = fixedRight

fixedRight_copy = fixedRight_copy.astype('float')
im_color = im_color.astype('float')
additionF = (fixedRight_copy+im_color)/2
addition = additionF.astype('uint8')

cv2.imshow('colored', addition)
cv2.waitKey(0)
cv2.destroyAllWindows()
