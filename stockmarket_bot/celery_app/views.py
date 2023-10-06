from django.shortcuts import render
from celery_app.tasks import print_statement

def trigger_task(request):
    print_statement.apply_async()
    # or simply print_statement() to execute immediately
    return render(request, 'trigger_task.html')
