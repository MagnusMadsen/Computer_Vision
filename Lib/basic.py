import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from imageOperations import patchMaker


# Indlæs billeder i henholdsvis BGR, gråtone og HSV 
bgr = cv2.imread("../images/shapes.jpg", cv2.IMREAD_COLOR)
gray = cv2.imread("../images/pills.jpg", cv2.IMREAD_GRAYSCALE)
bgr2 = cv2.imread("../images/foggy_forest.jpg")
hsv = cv2.cvtColor(bgr2, cv2.COLOR_BGR2HSV)

# Vis billeders opløsning
print(bgr.shape)
print(gray.shape)

# Nedskalering til anden opløsning
downscaleResolution = (300, 200)
downScaled = cv2.resize(gray, downscaleResolution, interpolation=cv2.INTER_LINEAR)

# Cropping
cropped = gray[50:100, 100:150] # Slice of x, slice of y
#patches = patchMaker(gray)

# Funktion til at hjælpe med at "eyeball" HSV thresholds
def hsv3Dscatter(hsv_img):
    h, s, v = cv2.split(hsv_img)

    pixel_colors = hsv_img.reshape((np.shape(hsv_img)[0]*np.shape(hsv_img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()

hsv3Dscatter(hsv)
'''
# Vis billeder
cv2.imshow('gray', gray)
cv2.imshow('down', downScaled)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
