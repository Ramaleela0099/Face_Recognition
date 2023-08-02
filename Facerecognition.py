import cv2
import face_recognition

input_movie = cv2.VideoCapture('0')
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

image = face_recognition.load_image_file('#give your sample photo')
face_encoding = face_recognition.face_encodings(image)[0]

known_faces = [
face_encoding,
]


face_locations = []
face_encodings = []
face_names = []
frame_number = []

while True:

    ret, frame = input_movie.read()
    frame_number += 1


    if not ret:
        break


    rgb_frame = frame[:, :, ::-1]


    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:

        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "#Give Your Name"
        face_names.append(name)


    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue


        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)


        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)


input_movie.release()
cv2.destroyAllWindows()
