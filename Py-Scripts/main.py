import face_recognition
import cv2
import numpy as np
import pandas as pd
import time
import sys
import beepy
import os
import urllib.request

def load_known_faces():
    known_face_encodings = []
    known_face_roll_no = []
    roll_record = {}

    df = pd.read_excel("students/students_db.xlsx")

    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    uploads_dir = os.path.join(current_directory, "public/assets/uploads/")
    
    if not os.path.exists(uploads_dir):
        print(f"Directory {uploads_dir} does not exist. Creating directory.")
        os.makedirs(uploads_dir)

    if os.path.exists(uploads_dir):
        print(f"Contents of {uploads_dir}: {os.listdir(uploads_dir)}")
    else:
        print(f"Failed to create directory {uploads_dir}")
        return known_face_encodings, known_face_roll_no, roll_record

    for _, row in df.iterrows():
        roll_no = row["roll_no"]
        name = row["name"]
        image_path = row["image"]
        roll_record[roll_no] = name

        try:
            image_file_path = os.path.join(uploads_dir, image_path)
            print(f"Attempting to load image from: {image_file_path}")
            if os.path.exists(image_file_path):
                student_image = face_recognition.load_image_file(image_file_path)
                student_face_encoding = face_recognition.face_encodings(student_image)[0]
                known_face_encodings.append(student_face_encoding)
                known_face_roll_no.append(roll_no)
            else:
                print(f"Error: File {image_file_path} does not exist")
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

    return known_face_encodings, known_face_roll_no, roll_record

def mark_attendance(name, roll_no):
    curr_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Attendance marked for {name} (Roll No: {roll_no}) at {curr_time}")
    # You can add your database storing logic here

def main():
    if len(sys.argv) < 4:
        print("Usage: main.py <subject_name> <class_id> <camera_source>")
        sys.exit(1)

    subject_name = sys.argv[1]
    class_id = sys.argv[2]
    camera_source = sys.argv[3]

    if camera_source == "empty":
        video_capture = cv2.VideoCapture(0)
    else:
        url = f"http://{camera_source}/shot.jpg"
        print("URL:", url)
        read = urllib.request.urlopen(url, timeout=20)

    known_face_encodings, known_face_roll_no, roll_record = load_known_faces()
    attendance_record = set()

    mark_attendance("Jayesh Dhobe", "811034")

    process_this_frame = True

    while True:
        try:
            if camera_source == "empty":
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to grab frame")
                    break
            else:
                imgResp = urllib.request.urlopen(url)
                imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
                frame = cv2.imdecode(imgNp, -1)

            frame = cv2.flip(frame, 1)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"

                    if True in matches:
                        first_match_index = matches.index(True)
                        roll_no = known_face_roll_no[first_match_index]
                        name = roll_record[roll_no]

                        if roll_no not in attendance_record:
                            attendance_record.add(roll_no)
                            beepy.beep(sound=1)
                            mark_attendance(name, roll_no)

                    face_names.append(name)

            process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            cv2.imshow("Video", frame)

        except Exception as e:
            print("Error:", e)
            continue

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if camera_source == "empty":
        video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
