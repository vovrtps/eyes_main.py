import cv2

eyes = cv2.CascadeClassifier("haarcascade_eye.xml")
video_capture = cv2.VideoCapture(0)

while True:
    flag, image = video_capture.read()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = eyes.detectMultiScale(gray_image, scaleFactor=2.5, minNeighbors=4)

    for x, y, w, h in faces:
        image[y:y + h, x//2:x + w * 3] = cv2.medianBlur(image[y:y + h, x//2:x + w * 3],35)

    cv2.imshow('eyes', image)
    key = cv2.waitKey(25)

    if key == ord('q'):
        break