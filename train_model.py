import os
import face_recognition
import pickle

from face_recognition import face_encodings


def train_model(image_folder):

    employee_encodings = {}

    #loop through the employee image folder
    for employee_folder in os.listdir(image_folder):
        employee_folder_path = os.path.join(image_folder, employee_folder)
        if os.path.isdir(employee_folder_path):
            print(f"Processing Image for {employee_folder}")
            employee_encodings[employee_folder] = []

            #loop through all images in each employee folder
            for image_file in os.listdir(employee_folder_path):
                image_path = os.path.join(employee_folder_path, image_file)
                image = face_recognition.load_image_file(image_path)

                #finding all encodings in an image
                face_encodings = face_recognition.face_encodings(image)

                if(face_encodings):
                    employee_encodings[employee_folder].append(face_encodings[0])

    with open('employee_encodings.pkl', 'wb') as f:
        pickle.dump(employee_encodings, f)
    print("Model trained and saved to 'employee_encodings.pkl'.")

train_model('employee_images')