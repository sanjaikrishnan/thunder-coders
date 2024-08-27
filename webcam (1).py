from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import cvlib as cv
                    
# Load the gender detection model
model = load_model('gender_detection.h5')

# Open the webcam
webcam = cv2.VideoCapture(0)
    
# Define the classes
classes = ['man', 'woman']

# Loop through the frames captured from the webcam
while webcam.isOpened():

    # Read a frame from the webcam
    status, frame = webcam.read()

    # Apply face detection
    faces, confidence = cv.detect_face(frame)

    # Initialize counters for men and women in this frame
    men_count = 0
    women_count = 0

    # Loop through the detected faces
    for idx, f in enumerate(faces):

        # Get the corner points of the face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the detected face region
        face_crop = np.copy(frame[startY:endY, startX:endX])

        # Skip small face detections
        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the face for the gender detection model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Apply gender detection on the face
        conf = model.predict(face_crop)[0]  # model.predict returns a 2D matrix, e.g., [[0.9999, 0.0001]]

        # Get the label with the highest confidence
        idx = np.argmax(conf)
        label = classes[idx]

        # Update the counter based on the prediction
        if label == 'man':
            men_count += 1
        else:
            women_count += 1

        # Prepare the label with confidence
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        # Set the position for the label on the image
        Y = startY - 10 if startY - 10 > 10 else startY + 10

        # Write the label and confidence above the face rectangle
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    # Display the counts of men and women on the frame
    count_label = f"Men: {men_count}, Women: {women_count}"
    cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)

    # Display the output frame
    cv2.imshow("Gender Detection and Counting", frame)

    # Press "Q" to stop the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam resources and close the display window
webcam.release()
cv2.destroyAllWindows()
