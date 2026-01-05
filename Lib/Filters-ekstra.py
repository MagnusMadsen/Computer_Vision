import cv2  
import numpy as np 
from matplotlib import pyplot as plt 

gray = cv2.imread("../images/Daniel_billeder/IMG_0331.jpeg", 0)
bgr = cv2.imread("../images/Daniel_billeder/IMG_0331.jpeg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

gauss = cv2.GaussianBlur(bgr, (11, 11), 0)
bilat = cv2.bilateralFilter(gray, 5, sigmaColor=75, sigmaSpace=75)


def compareEdges(filteredImg):
    sobelx = cv2.Sobel(filteredImg, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(filteredImg, cv2.CV_64F, 0, 1, ksize=5)
    canny = cv2.Canny(filteredImg, 100, 200)
    laplacian = cv2.Laplacian(filteredImg, cv2.CV_64F)

    # Konverter til uint8 så de kan vises korrekt
    sobelx = cv2.convertScaleAbs(sobelx)
    sobely = cv2.convertScaleAbs(sobely)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Sørg for ens kanaler (grayscale)
    if len(sobelx.shape) == 2:
        sobelx = cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
        sobely = cv2.cvtColor(sobely, cv2.COLOR_GRAY2BGR)
        laplacian = cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
        canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    top = np.hstack((sobelx, sobely))
    bottom = np.hstack((laplacian, canny))
    combined = np.vstack((top, bottom))

    # Tilføj labels med baggrund (så de altid er synlige)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    text_color = (255, 255, 255)   # Hvid tekst
    bg_color = (0, 0, 0)           # Sort baggrund
    thickness = 2
    padding = 6

    h, w, _ = sobelx.shape

    def draw_label(img, text, x, y):
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.rectangle(
            img,
            (x - padding, y - th - padding),
            (x + tw + padding, y + padding),
            bg_color,
            -1
        )
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

    draw_label(combined, "Sobel X", 10, 30)
    draw_label(combined, "Sobel Y", w + 10, 30)
    draw_label(combined, "Laplace", 10, h + 30)
    draw_label(combined, "Canny", w + 10, h + 30)

    cv2.imshow("Edge comparison (Sobel X | Sobel Y | Laplace | Canny)", combined)
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


# NEW FUNCTION: interactiveEdgeTuning
def interactiveEdgeTuning(grayimg):
    cv2.namedWindow("Interactive Edge Tuning")

    def nothing(x):
        pass

    # Trackbars
    cv2.createTrackbar("Canny low", "Interactive Edge Tuning", 50, 255, nothing)
    cv2.createTrackbar("Canny high", "Interactive Edge Tuning", 150, 255, nothing)
    cv2.createTrackbar("Gaussian k", "Interactive Edge Tuning", 5, 31, nothing)
    cv2.createTrackbar("Filter", "Interactive Edge Tuning", 0, 2, nothing)
    # Filter: 0=None, 1=Gaussian, 2=Bilateral

    while True:
        low = cv2.getTrackbarPos("Canny low", "Interactive Edge Tuning")
        high = cv2.getTrackbarPos("Canny high", "Interactive Edge Tuning")
        k = cv2.getTrackbarPos("Gaussian k", "Interactive Edge Tuning")
        filt = cv2.getTrackbarPos("Filter", "Interactive Edge Tuning")

        if k % 2 == 0:
            k += 1
        if k < 1:
            k = 1

        work = grayimg.copy()

        if filt == 1:
            work = cv2.GaussianBlur(work, (k, k), 0)
        elif filt == 2:
            work = cv2.bilateralFilter(work, 9, 75, 75)

        edges = cv2.Canny(work, low, high)

        cv2.imshow("Interactive Edge Tuning", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

compareThresholds(bilat)
#interactiveEdgeTuning(gray)
#blobDetection(gray)
#compareEdges(bilat)
#hueEdges(hsv)
#contourDetection(gray)
