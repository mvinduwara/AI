import cv2
from insightface.app import FaceAnalysis

faceAnalysis = FaceAnalysis(name="buffalo_l")  # https://github.com/deepinsight/insightface/releases
faceAnalysis.prepare(ctx_id=0, det_size=(640, 640))

Image1 = cv2.imread("person-01.png")
# Image2 = cv2.imread("person-02.png")

faces = faceAnalysis.get(Image1)
face = faces[0]

cv2.imshow("Image", Image1)
cv2.waitKey(0)
