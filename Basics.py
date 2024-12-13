import cv2
import numpy as np
import face_recognition

imgLana = face_recognition.load_image_file('ImageBasic/lana1.jpg')
imgLana = cv2.cvtColor(imgLana, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/lana2.webp')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgLana)[0]
encodeLana = face_recognition.face_encodings(imgLana)[0]
cv2.rectangle(imgLana, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 255, 0), 2)

results = face_recognition.compare_faces([encodeLana], encodeTest)
faceDis = face_recognition.face_distance([encodeLana], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

cv2.imshow('Lana', imgLana)
cv2.imshow('Test1', imgTest)
cv2.waitKey(0)