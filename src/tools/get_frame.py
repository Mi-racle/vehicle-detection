import cv2

cap_in = cv2.VideoCapture('D:/xxs-signs/vehicle-detection/resources/traffic2.mp4')
ret, frame = cap_in.read()
cv2.imwrite('D:/xxs-signs/vehicle-detection/resources/image.png', frame)
cap_in.release()
