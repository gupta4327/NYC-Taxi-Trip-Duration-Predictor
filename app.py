from flask import Flask, render_template, request
from predictor import TripDurationPredictor


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

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
    print(form_data)
    duration = TripDurationPredictor()
    pred = duration.predict_duration(form_data)
    pred_time = round(pred/60,2)

    return render_template('predict.html',pred_time = pred_time)


if __name__ == "__main__":
	
	app.run(host='0.0.0.0')		
