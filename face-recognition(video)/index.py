import cv2

model = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

originalImage = cv2.imread('image.png')

# cv2.imshow('Original Image', originalImage)
# greyScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)
greyScaleImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)     # opencv color schemes are BGR, not RGB

faces = model.detectMultiScale(greyScaleImage,scaleFactor=1.05, minNeighbors=4)

for(x,y,w,h) in faces:
    cv2.rectangle(originalImage, (x,y), (x+w,y+h), (0,0,255), 2)

cv2.imshow('Original Image', originalImage)
cv2.waitKey(0)


