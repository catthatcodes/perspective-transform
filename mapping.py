import cv2

image = cv2.imread("box1.png")

dim = (700, 700)
new = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
 
cv2.imshow("image", image)
cv2.imshow("new", new)

if cv2.waitKey(0) & 0xff == ord('q'):
	cv2.destroyAllWindows()