import cv2
# our image
img_file = 'cars in road.jpg'
video = cv2.VideoCapture('cars.mp4')
# our pre-trained car classifier
car_tracker_file = 'car_detector.xml'
# our pre-trained pedesitrain classifier
pedestrain_tracker_file = 'pedestrain.xml'


# create opencv image

img = cv2.imread(img_file)

# create classifier for car

car_tracker = cv2.CascadeClassifier(car_tracker_file)

# create classifier for pedestrain
pedestrain_tracker = cv2.CascadeClassifier(pedestrain_tracker_file)

# run for ever

while True:
    # read the current frame:
    (read_successful, frame) = video.read()
    # safe coding:
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # car and pedestrains detect

    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrain = pedestrain_tracker.detectMultiScale(grayscaled_frame)

    # drawing rectangle for pedestrains:
    for(x, y, w, h) in pedestrain:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 2)

     # drawing rectangle for cars:
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+2), (x+w, y+h), (225, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 225), 2)

    # Display the images in the video
    cv2.imshow("Self driving car", frame)

    # don't autoclose wait for  key press

    key = cv2.waitKey(1)
# stop if Q key is presses
    if key == 81 or key == 113:
        break

# Release the VideoCapture object
video.release()

"""
# black and white the images

black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# create classifier

car_tracker = cv2.CascadeClassifier(classifier_file)

# car detect

cars = car_tracker.detectMultiScale(black_n_white)


# draw rectangles in around the cars:

for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 225, 2)

print(cars)
# Display the image

cv2.imshow("car detector", img)

# don't autoclose wait for  key press

cv2.waitKey()

print("lets go")
"""
