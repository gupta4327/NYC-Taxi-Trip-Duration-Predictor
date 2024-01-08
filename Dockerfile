FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

COPY ./deployement/app.py /app/app.py
COPY bestmodel.pkl /app/bestmodel.pkl
COPY loc_kmeans.pkl /app/loc_kmenas.pkl  
COPY ./deployement/features.yaml /app/features.yaml
COPY ./deployement/predictor.py /app/predictor.py
COPY ./deployement/build_features.py /app/build_features.py
COPY ./deployement/requirements.txt /app/requirements.txt
COPY ./deployement/logger.py /app/logger.py
COPY ./deployement/static /app/static
COPY ./deployement/templates /app/templates

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Corrected CMD syntax with square brackets
CMD ["python", "app.py"]
