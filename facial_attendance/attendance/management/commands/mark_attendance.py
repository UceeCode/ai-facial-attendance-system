import cv2
import dlib
import pickle
from django.core.management.base import BaseCommand
from attendance.models import Employee, Attendance
from datetime import date
import numpy as np
import os
import time  # Import the time module

class Command(BaseCommand):
    help = "Run facial recognition and mark attendance using dlib"

    def handle(self, *args, **kwargs):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.stdout.write(self.style.ERROR("❌ Cannot open webcam"))
            return
        cap.set(3, 1280)
        cap.set(4, 720)

        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(script_dir, "shape_predictor_68_face_landmarks.dat")
        face_rec_model_path = os.path.join(script_dir, "dlib_face_recognition_resnet_model_v1.dat")

        # Load dlib face detector and shape predictor
        detector = dlib.get_frontal_face_detector()
        try:
            predictor = dlib.shape_predictor(shape_predictor_path)
        except RuntimeError as e:
            self.stdout.write(self.style.ERROR(f"❌ Error loading shape predictor: {e}"))
            return
        try:
            face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
        except RuntimeError as e:
            self.stdout.write(self.style.ERROR(f"❌ Error loading face recognition model: {e}"))
            return

        # Load all employee face encodings from the database
        employees = Employee.objects.all()
        employee_encodings = {}
        employee_details = {}

        for employee in employees:
            encoding_array = pickle.loads(employee.face_encoding)
            employee_encodings[employee.name] = np.array(encoding_array)
            employee_details[employee.name] = employee.email

        known_encodings = list(employee_encodings.values())
        known_names = list(employee_encodings.keys())

        self.stdout.write("🔍 Starting webcam for facial recognition using dlib... Press 'q' to quit.")

        last_marked_name = None
        last_marked_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                self.stdout.write(self.style.ERROR("❌ Failed to read frame from webcam."))
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray_frame)

            face_encodings_current_frame = []
            face_locations = []

            for face in faces:
                face_locations.append((face.top(), face.right(), face.bottom(), face.left()))
                landmarks = predictor(gray_frame, face)
                face_encoding = np.array(face_rec_model.compute_face_descriptor(frame, landmarks, 1))
                face_encodings_current_frame.append(face_encoding)

            for face_encoding, face_location in zip(face_encodings_current_frame, face_locations):
                name = "Unknown"
                email = ""
                min_distance = 1.0  # Initialize with a large value

                for i, known_encoding in enumerate(known_encodings):
                    distance = np.linalg.norm(face_encoding - known_encoding)
                    tolerance = 0.6  # You can adjust this threshold

                    if distance < tolerance and distance < min_distance:
                        min_distance = distance
                        name = known_names[i]
                        email = employee_details[name]

                if name != "Unknown":
                    employee = Employee.objects.get(name=name)
                    if not Attendance.objects.filter(employee=employee, date=date.today()).exists():
                        Attendance.objects.create(employee=employee)
                        self.stdout.write(self.style.SUCCESS(f"✅ Attendance marked for {name}"))
                        last_marked_name = name  # Update last marked name
                        last_marked_time = time.time()  # Update last marked time

                top, right, bottom, left = face_location
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                info_text = f"{name} - {email}" if name != "Unknown" else name
                cv2.putText(frame, info_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if name != "Unknown":
                    tick_center = (right - 20, top + 20)
                    cv2.circle(frame, tick_center, 10, (0, 255, 0), -1)
                    cv2.line(frame, (tick_center[0] - 3, tick_center[1]),
                             (tick_center[0], tick_center[1] + 3), (255, 255, 255), 2)
                    cv2.line(frame, (tick_center[0], tick_center[1] + 3),
                             (tick_center[0] + 5, tick_center[1] - 5), (255, 255, 255), 2)

            if last_marked_name and time.time() - last_marked_time < 5:
                banner_text = f"✅ Attendance recorded for {last_marked_name}"
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 128, 0), -1)
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.putText(frame, banner_text, (30, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                last_marked_name = None # Reset the banner

            cv2.imshow("Facial Attendance System (dlib)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()