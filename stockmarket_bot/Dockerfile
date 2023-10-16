# Use the official Django image
FROM python:3.9-slim

# Set environment variables for Django
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE stockmarket_bot.settings

# Copy the current directory contents into the container at /app
WORKDIR /app
COPY requirements.txt /app

# Set the working directory to /app

RUN apt-get update && apt-get install -y gcc libffi-dev libssl-dev python3-dev libpq-dev && apt-get clean
RUN pip install -r requirements.txt

COPY . /app

# Run the Django server
CMD ["gunicorn", "myproject.wsgi:application", "--bind", "0.0.0.0:8000"]
