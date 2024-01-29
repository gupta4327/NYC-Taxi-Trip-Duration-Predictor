from flask import Flask, render_template, request
from predictor import TripDurationPredictor
from logger import infologger

app = Flask(__name__)


@app.route('/')
def index():
    
    try:
        return render_template('index.html')
    
    except Exception as e:
         infologger.info(f'Home page has not been loaded : {e}')

@app.route('/predict', methods=['POST'])
def predict():

    form_data = dict()
    form_data['vendor_id'] = request.form.get('vendor_id', type=int)
    form_data['pickup_latitude'] = request.form.get('pickup_latitude',type=float)
    form_data['pickup_longitude'] = request.form.get('pickup_longitude',type=float)
    form_data['dropoff_latitude'] = request.form.get('dropoff_latitude',type=float)
    form_data['dropoff_longitude'] = request.form.get('dropoff_longitude',type=float)
    form_data['pickup_datetime'] = request.form.get('pickup_datetime')
    form_data['store_and_fwd_flag'] = request.form.get('store_and_fwd_flag')
    try:
        duration = TripDurationPredictor()
        pred = duration.predict_duration(form_data)[0].item()
        pred_time = round(pred/60,2)
        return render_template('predict.html',pred_time = pred_time)
    
    except Exception as e:
         infologger.info(f'Failed with error : {e}')
         print(f'Failed with error : {e}')


if __name__ == "__main__":
     
     try:
         app.run(host='0.0.0.0',port=5000)
         
         
     except Exception as e:
        infologger.info(f'Application failure : {e}')		
