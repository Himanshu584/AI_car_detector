import cv2
import numpy as np
import keyboard

# #cars image
# user_input = input("please enter image file:\n >>>" )
# img_file = user_input

#Video file
video_file = input("please enter Video file:\n >>>")

# Read through video file
video = cv2.VideoCapture(video_file)

#pre trained car detector haar file
car_classifier_file =  r'C:\\Users\\Asus\\Documents\\car_detector.xml'

# create car classifier 
car_tracker = cv2.CascadeClassifier(car_classifier_file)

# run forever untill video ends or manually stopped
while True:
    # read current frame
    (read_successful, frame) = video.read()

    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # detect cars 
    cars = car_tracker.detectMultiScale(grayscaled_frame)

    #print(cars)    #--> uncomment to see matrix values of cars detected

    # draw rectangles around cars
    for (x,y,w,h) in cars:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255), 2)

    #display video
    cv2.imshow('Galvic Car Detector', frame)
    
    if keyboard.is_pressed(" "):
        break
    else:
        # prevent autoclose untill key is pressed 
        cv2.waitKey(1)


cv2.destroyAllWindows()


print('code completed!')