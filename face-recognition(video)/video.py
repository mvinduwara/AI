import cv2

model = face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture('video.mp4')

while True:
    b,Image = video.read()

    if b:
        greyScaleImage = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)  

        faces = model.detectMultiScale(greyScaleImage,scaleFactor=1.2, minNeighbors=9)

        for(x,y,w,h) in faces:
            cv2.rectangle(Image, (x,y), (x+w,y+h), (0,0,255), 2)

            cv2.imshow('Original Video', Image)
            cv2.waitKey(1)




             

