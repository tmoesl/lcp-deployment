FROM python:3.9-slim

WORKDIR /app

# Copy only the requirements file
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the frontend application file
COPY main.py .

CMD ["streamlit", "run", "main.py"]