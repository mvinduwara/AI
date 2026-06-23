import cv2

d = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


originalImage = cv2.imread('image1.png')
x = cv2.cvtColor(originalImage, cv2.COLOR_RGB2GRAY)
# x = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

face = d.detectMultiScale(x, scaleFactor=1.1, minNeighbors=5)

for(x,y,w,h) in face:
    cv2.rectangle(originalImage , (x,y) , (x+w,y+h) , (0,0,255) , 2)

cv2.imshow('image', originalImage)
cv2.waitKey(0)

# print(cv2.__version__)