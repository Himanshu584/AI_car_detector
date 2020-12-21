import cv2

#cars image
user_input = input("please enter image file:\n >>>" )
img_file = user_input

#pre trained car detector haar file
car_classifier_file =  r'C:\\Users\\Asus\\Documents\\car_detector.xml'

# create car classifier 
car_tracker = cv2.CascadeClassifier(car_classifier_file)

#read through image
img = cv2.imread(img_file)

# conert image to grayscale ( important for classifier detection )
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect cars 
cars = car_tracker.detectMultiScale(black_n_white)
# print(cars)

# # draw rectangles around cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h),(0,0,255), 2)

#display image
cv2.imshow('Galvic Car Detector', img)

# prevent autoclose untill key is pressed 
cv2.waitKey()

cv2.destroyAllWindows()
# code ran without errors
print('code completed!')
