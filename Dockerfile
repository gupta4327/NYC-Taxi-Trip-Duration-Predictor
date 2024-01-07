From python:3.10-slim

# Set the working directory to /app
WORKDIR /app

copy app.py /app/app.py
copy bestmodel.pkl /app/bestmodel.pkl
copy loc_kmeans.pkl /app/loc_kmenas.pkl
copy params.yaml /app/params.yaml
copy predictor.py /app/predictor.py
copy ./src/features/build_features.py /app/build_features.py
copy requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

cmd ['python' 'app.py']



