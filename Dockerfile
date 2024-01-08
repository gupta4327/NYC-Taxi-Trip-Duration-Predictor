FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

COPY app.py /app/app.py
COPY bestmodel.pkl /app/bestmodel.pkl
COPY loc_kmeans.pkl /app/loc_kmenas.pkl  
COPY params.yaml /app/params.yaml
COPY predictor.py /app/predictor.py
COPY ./src/features/build_features.py /app/build_features.py
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Corrected CMD syntax with square brackets
CMD ["python", "app.py"]
