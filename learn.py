import cv2

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the pre-trained facial landmark model (you'll need to download the shape_predictor_68_face_landmarks.dat file)
landmark_predictor = cv2.face_LandmarkPredictor('shape_predictor_68_face_landmarks.dat')

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and landmarks (eyes and mouth)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Detect landmarks on the face
        landmarks = landmark_predictor(gray, faces[0])

        # Draw landmarks (eyes and mouth)
        for n in range(0, 68):
            x_landmark, y_landmark = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x_landmark, y_landmark), 1, (0, 0, 255), -1)

    # Display the number of faces detected
    num_faces = len(faces)
    cv2.putText(frame, f'Faces detected: {num_faces}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get the current timestamp and display it
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with detected faces and landmarks
    cv2.imshow('Face Detection', frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
