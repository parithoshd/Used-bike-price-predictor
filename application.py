from flask import Flask , render_template ,request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
bikes = pd.read_csv("E:/project_bike/Cleaned_data.csv")

model = pickle.load(open('E:/project_bike/LinearRegressionModel.pkl', 'rb'))

@app.route('/')
def index():
    brands = sorted(bikes['brand'].unique())
    bikenames = sorted(bikes['bike_name'].unique())
    owners = sorted(bikes['owner'].unique())
    ages = sorted(bikes['age'].unique())
    powers = sorted(bikes['power'].unique())
    brands.insert(0,"Select Company")
    return render_template('index.html', brands=brands, bikenames=bikenames,
                           owners=owners, ages=ages, powers=powers)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    bikename = request.form.get('bikename')
    brand = request.form.get('brand')
    kms_driven = float(request.form.get('kms_driven'))
    age = float(request.form.get('age'))
    power = float(request.form.get('power'))
    owner = request.form.get('owner')
    print(kms_driven)
    prediction = model.predict(pd.DataFrame([[bikename, kms_driven, owner, age, power, brand]], columns=[
                               'bike_name', 'kms_driven', 'owner', 'age', 'power', 'brand']))
    print(prediction)
    return str(np.round(prediction[0], 2))

if __name__=='__main__':
    app.run(debug=True)