import cv2  
import numpy as np 
from matplotlib import pyplot as plt 

gray = cv2.imread("../images/pills.jpg", 0)
bgr = cv2.imread("../images/zebra.jpg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gauss = cv2.GaussianBlur(bgr, (11, 11), 0)
bilat = cv2.bilateralFilter(gray, 5, sigmaColor=75, sigmaSpace=75)


def compareEdges(filteredImg):
    sobelx = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    sobely = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelxy = cv2.Sobel(src=filteredImg, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
    canny = cv2.Canny(image=filteredImg, threshold1=100, threshold2=200)
    laplacian = cv2.convertScaleAbs(cv2.Laplacian(filteredImg, cv2.CV_64F))

    cv2.imshow("Sobel X", sobelx)
    cv2.imshow("Sobel Y", sobely)
    cv2.imshow("Sobel XY", sobelxy)
    cv2.imshow("Laplace", laplacian)
    cv2.imshow("Canny", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def compareThresholds(blurred_grayimg):
    th1 = cv2.adaptiveThreshold(blurred_grayimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th2 = cv2.adaptiveThreshold(blurred_grayimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, th3 = cv2.threshold(blurred_grayimg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    titles = ["Original", "Otsu", "Adaptive", "Gaussian"]
    images = [blurred_grayimg, th1, th2, th3]

    for i in range(len(images)):
        plt.subplot(2,2,i+1), plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()
    


def hueEdges(hsvimg):
    shift = 25;
    h, s, v = cv2.split(hsvimg)
    shiftedHue = h.copy()

    height = shiftedHue.shape[0]
    width = shiftedHue.shape[1]
    for y in range(0, height):
        for x in range(0, width):
            shiftedHue[y, x] = (h[y, x] + shift)%180

    canny = cv2.Canny(shiftedHue, 150, 255);

    cv2.imshow("Canny on shifted hue", canny)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def contourDetection(grayimg):
    grayimg = cv2.bilateralFilter(grayimg, 9, sigmaColor=75, sigmaSpace=75)
    ret, thresh = cv2.threshold(grayimg, 100, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    grayCopy = grayimg.copy()
    cv2.drawContours(grayCopy, contours, -1, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Contours", grayCopy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blobDetection(grayimg):
    parameters = cv2.SimpleBlobDetector_Params()
    
    parameters.filterByArea = True
    parameters.minArea = 10 
    parameters.filterByCircularity = True 
    parameters.minCircularity = 0.1

    detector = cv2.SimpleBlobDetector_create(parameters)

    keypoints = detector.detect(grayimg)
    imageWithKeypoints = cv2.drawKeypoints(grayimg, 
                                           keypoints, 
                                           np.array([]), 
                                           (0,0,255), 
                                           cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
                                           )
    cv2.imshow("Blobs", imageWithKeypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def showComparison():
    cv2.imshow("org", bgr)
    cv2.imshow("gauss", gauss)
    cv2.imshow("bilat", bilat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

compareThresholds(bilat)
#blobDetection(gray)
#compareEdges(bilat)
#hueEdges(hsv)
#contourDetection(gray)
