import cv2

i = cv2.imread('image1.png')
cv2.imshow('image', i)
cv2.waitKey(0)

# print(cv2.__version__)