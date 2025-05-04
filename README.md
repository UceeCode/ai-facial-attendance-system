# AI Facial Attendance System

## Overview

This project implements an AI-powered facial attendance system using Python, OpenCV, Dlib, and Django. It captures video from a webcam, detects faces, recognizes employees, and automatically marks their attendance in a database.

## Features

* **Real-time Facial Detection:** Utilizes Dlib's robust face detection algorithm.
* **Facial Recognition:** Employs Dlib's deep learning model for generating and comparing facial embeddings.
* **Employee Registration:** Allows for storing employee names, email addresses, and their facial encodings in a Django database.
* **Automated Attendance Marking:** Automatically records the attendance of recognized employees based on the current date.
* **Webcam Integration:** Captures video feed directly from the user's webcam.
* **Clear User Interface:** Displays the video feed with bounding boxes around detected faces and the names of recognized employees.
* **Attendance Logging:** Records attendance details (employee, date, time - implicitly by the time of recognition) in the Django database.
* **Visual Feedback:** Provides on-screen visual cues (green box and tick) for recognized and marked employees, along with a temporary banner message.

## Technologies Used

* **Python:** The primary programming language.
* **OpenCV (cv2):** For real-time video capture and image processing.
* **Dlib:** A modern C++ toolkit containing machine learning algorithms and tools for creating complex real-world applications. Used here for face detection and facial recognition.
* **NumPy:** For efficient numerical operations, especially with facial embeddings.
* **Pickle:** For serializing and deserializing Python objects (used for storing facial encodings in the database).
* **Django:** A high-level Python web framework used for building the backend, managing the database (models for Employees and Attendance), and creating management commands for running the facial recognition system.
* **MySQL (or your configured Django database):** For storing employee information and attendance records.

## Installation

1.  **Clone the repository (if you have one):**
    ```bash
    git clone <repository_url>
    cd <project_directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On macOS and Linux
    .venv\Scripts\activate  # On Windows
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    (You might need to create a `requirements.txt` file with the following dependencies if it doesn't exist):
    ```
    opencv-python
    dlib
    numpy
    Django
    mysqlclient
    ```

4.  **Download Dlib's pre-trained models:**
    * Download `shape_predictor_68_face_landmarks.dat.bz2` from [dlib-models](https://github.com/davisking/dlib-models). Extract `shape_predictor_68_face_landmarks.dat` and place it in the same directory as your `manage.py` file or within the `attendance/management/commands/` directory.
    * Download `dlib_face_recognition_resnet_model_v1.dat.bz2` from [dlib-models](https://github.com/davisking/dlib-models). Extract `dlib_face_recognition_resnet_model_v1.dat` and place it in the same directory as your `manage.py` file or within the `attendance/management/commands/` directory.

5.  **Configure your Django database:**
    * Edit the `DATABASES` setting in your project's `settings.py` file to match your MySQL (or other database) configuration.
    * Run migrations to create the database tables:
        ```bash
        python manage.py migrate
        ```

## Setup and Usage

1.  **Run the facial attendance system:**
    Navigate to your project's root directory and run the custom Django management command:
    ```bash
    python manage.py mark_attendance
    ```
    This command will:
    * Start capturing video from your webcam.
    * Detect faces in the video frames.
    * Compare the detected faces against the facial encodings stored in your database.
    * If a match is found, it will mark the attendance for that employee for the current date (if not already marked) and display visual feedback.
    * Press `q` on the video window to quit the application.
