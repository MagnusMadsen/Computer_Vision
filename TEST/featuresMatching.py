import cv2  
from matplotlib import pyplot as plt 


# De 2 billeder vi kommer til at bruge til sammenligning
gray = cv2.imread("../images/boat.jpg", 0)
grayComparison = cv2.imread("../images/castle.jpg", 0)

# Background subtraction: https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html


bgr = cv2.imread("../images/zebra.jpg")
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

# Ikke nødvendigt i keypoint detection
gauss = cv2.GaussianBlur(gray, (11, 11), 0)
bilat = cv2.bilateralFilter(gray, 11, sigmaColor=75, sigmaSpace=75)


def orbKeypoints(grayimg):
    orb = cv2.ORB.create()
    keypoints, destination = orb.detectAndCompute(grayimg, None)
    return keypoints, destination
    

def siftKeypoints(grayimg):
    sift = cv2.SIFT.create()
    keypoints, destination = sift.detectAndCompute(grayimg, None)
    return keypoints, destination

# Kun for ORB keypoints
def bruteforceMatching(img1, img2, keypoints1, keypoints2, description1, description2):
    # Hamming distance som metric er nødvendig for orb keypoints
    brute = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = brute.match(description1, description2)
    # Viser kun 10 bedste matches, ændre matches[:10] for flere/færre
    imageWithMatches = cv2.drawMatches(img1, keypoints1, 
                                       img2, keypoints2, 
                                       matches[:10], None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def knnDistanceMatch(img1, img2, keypoints1, keypoints2, description1, description2):
    brute = cv2.BFMatcher()
    matches = brute.knnMatch(description1, description2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, 
                                          img2, keypoints2,
                                          good, None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(imageWithMatches), plt.show()


def flannKnnMatching(img1, img2, keypoints1, keypoints2, description1, description2):
    index = dict(algorithm = 1, trees = 5)
    search = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(indexParams=index, searchParams=search)
    matches = flann.knnMatch(description1, description2, k=2)
    matchesMask = [[0,0] for i in range(len(matches))]
    
    for i, (m,n) in enumerate(matches):
        if m.distance < 0.75 * n.distance:
            matchesMask[i] = [1,0]

    drawParams = dict(matchColor = (0, 255, 0),
                     singlePointColor = (255, 0, 0),
                     matchesMask = matchesMask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    plt.imshow(imageWithMatches,),plt.show()


k1, d1 = siftKeypoints(gray)
k2, d2 = siftKeypoints(grayComparison)

flannKnnMatching(gray, grayComparison, k1, k2, d1, d2)
#knnDistanceMatch(gray, grayComparison, k1, k2, d1, d2)
