import cv2
from insightface.app import FaceAnalysis
import numpy 

faceAnalysis = FaceAnalysis(name="buffalo_l")                       # https://github.com/deepinsight/insightface/releases
faceAnalysis.prepare(ctx_id=0, det_size=(640, 640))

Image1 = cv2.imread("person-01.png")
Image2 = cv2.imread("person-02.png")

faces = faceAnalysis.get(Image1)
face = faces[0]

facesA = faceAnalysis.get(Image2)
faceA = facesA[0]

code1 = face.embedding
code2 = faceA.embedding

print(code1)
print(code2)

x1,y1,x2,y2 = map(int,face.bbox)
cv2.rectangle(Image1,(x1,y1),(x2,y2),(0,0,255),2)

distance = numpy.linalg.norm(code1-code2)

print(distance)

cv2.imshow("Image", Image1)
cv2.waitKey(0)
