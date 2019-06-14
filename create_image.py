import cv2
import sys
# Load the Haar cascades
face_cascade = cv2.CascadeClassifier('/home/nguyda/Documents/insight/Insight_Project_Framework/Insight_Project_Framework/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/home/nguyda/Documents/insight/Insight_Project_Framework/Insight_Project_Framework/haarcascade_eye.xml')

# Define function that will do detection
def detect(gray, frame):
    """ Input = image
        Output = Image with rectangle detection boxes
    """
    # Now get the tuples that detect the faces using the above cascade
    kernel_size = 1.3
    num_neighbors = 5
    # kernel_size is the size of the image reduced when applying the detection
    # num_neighbors is the num of neighbors after which we accept that is a face
    faces = face_cascade.detectMultiScale(gray, kernel_size, num_neighbors)

    # faces are the tuples of the 4 numbers
    # x,y => upperleft corner coordinates of face
    # width(w) of rectangle in the face
    # height(h) of rectangle in the face

    # Now iterate over the faces and detect eyes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Arguements => image, top-left coordinates, bottomright coordinates, color, rectangle border thickness

        # we now need two region of interests(ROI) grey and color for eyes one to detect and another to draw rectangle
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # Detect eyes now
        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # Now draw rectangle over the eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


    return frame
filepath = r'/home/nguyda/Documents/insight/Insight_Project_Framework/data/raw/2.jpg'
img = cv2.imread(filepath)
cv2.imshow("before", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Call the detect function with grey image and colored frame
canvas = detect(gray, img)
# Show the image in the screen
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imshow("canvas", canvas)

cv2.waitKey(0)
if input("are you sure? (y/n)") != "y":
    exit()