from django.shortcuts import render
from attendance.models import Attendance

def dashboard(request):
    attendance_records = Attendance.objects.all()
    return render(request, 'attendance/dashboard.html', {'attendance_records': attendance_records})
