import cv2  
from matplotlib import pyplot as plt 


# De 2 billeder vi kommer til at bruge til sammenligning
gray = cv2.imread("../images/Daniel_face.png", 0)

cap = cv2.VideoCapture("../images/Convertet.mp4")

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
        if m.distance < 0.85 * n.distance:
            matchesMask[i] = [1,0]

    drawParams = dict(matchColor = (0, 255, 0),
                     singlePointColor = (255, 0, 0),
                     matchesMask = matchesMask,
                     flags = cv2.DrawMatchesFlags_DEFAULT)
    imageWithMatches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, matches, None, **drawParams)
    plt.imshow(imageWithMatches,),plt.show()


gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
k1, d1 = orbKeypoints(gray_small)

orb = cv2.ORB.create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame_small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    grayComparison = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)

    k2, d2 = orb.detectAndCompute(grayComparison, None)
    if d2 is None:
        continue

    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)[:30]

    matchedFrame = cv2.drawMatches(
        gray_small, k1,
        grayComparison, k2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imshow("Live Feature Matching (~25 FPS) - press q to quit", matchedFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()