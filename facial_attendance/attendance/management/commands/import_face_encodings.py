import os
import sys
import django
import pickle

# Add the path to your Django project's root directory
PROJECT_ROOT = '/Users/uchenna/Desktop/Workspace/facial_attendance_system/facial_attendance'  # Adjust this path if necessary
sys.path.append(PROJECT_ROOT)

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facial_attendance.settings')

# Initialize Django
django.setup()

from attendance.models import Employee

# Load the saved face encodings
PICKLE_FILE_PATH = '/Users/uchenna/Desktop/Workspace/facial_attendance_system/employee_encodings.pkl'

# Define a mapping of employee names (from pickle file) to their actual emails
EMPLOYEE_EMAILS = {
    "Uchenna": "writefranklyn@gmail.com",
    "Olivia": "osujiolivia06@gmail.com",
    "Anita": "writeanita.n@gmail.com",
    "Onyinyechi": "victoria.n@gmail.com"
}

try:
    with open(PICKLE_FILE_PATH, 'rb') as f:
        employee_encodings = pickle.load(f)

    # Insert face encodings into the database
    for employee_name, encodings in employee_encodings.items():
        if employee_name in EMPLOYEE_EMAILS:
            employee_email = EMPLOYEE_EMAILS[employee_name]
            try:
                employee, created = Employee.objects.get_or_create(
                    name=employee_name,
                    email=employee_email
                )
                if not created:
                    employee.face_encoding = pickle.dumps(encodings[0])
                    employee.save()
                    print(f"Updated face encoding for existing employee: {employee_name} ({employee_email})")
                else:
                    employee.face_encoding = pickle.dumps(encodings[0])
                    employee.save()
                    print(f"Added face encoding for new employee: {employee_name} ({employee_email})")
            except django.db.utils.IntegrityError:
                print(f"Skipping duplicate email: {employee_email} for {employee_name}")
            except Exception as e:
                print(f"Error processing {employee_name} ({employee_email}): {e}")
        else:
            print(f"Warning: Email not found for employee: {employee_name}")

    print("Employee face encodings processing complete.")

except FileNotFoundError:
    print(f"Error: '{PICKLE_FILE_PATH}' not found. Make sure the file is in the correct location.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")