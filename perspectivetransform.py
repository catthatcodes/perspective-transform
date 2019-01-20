import imutils
import numpy as np
import cv2

image = cv2.imread("ex7.jpg")
ratio = image.shape[0] / 200.0
org = image.copy()
image = imutils.resize(image, height = 200)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edges = cv2.Canny(gray, 30, 200)

cnts = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse = True)[:4]
screenCnt = None
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.015*peri, True)
	
	if(len(approx) == 4):
		screenCnt = approx
		break

pts = screenCnt.reshape(4,2)
rect = np.zeros((4,2), dtype = "float32")

s = pts.sum(axis = 1)
rect[0] = pts[np.argmin(s)]
rect[2] = pts[np.argmax(s)]

diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

rect*=ratio

(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0])**2) + ((br[1] - br[1])**2))
widthB = np.sqrt(((tr[0] - tl[0])**2) + ((tr[1] - tl[1])**2))

heightA = np.sqrt(((tr[0] - br[0])**2) + ((tr[1] - br[1])**2))
heightB = np.sqrt(((tl[0] - bl[0])**2) + ((tl[1] - bl[1])**2))

maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
		[0,0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(org, M, (maxWidth, maxHeight))
cv2.imshow("warp", warp)
cv2.imshow("image", image)
cv2.imshow("edges", edges)

if cv2.waitKey(0) & 0xff == ord('q'):
	cv2.destroyAllWindows()
	