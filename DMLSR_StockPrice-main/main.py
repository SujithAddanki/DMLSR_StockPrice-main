from flask import Flask, render_template, request, jsonify
import requests
from my_model import predict_price, plot_predictions

app = Flask(__name__)

ALPHA_VANTAGE_API_KEY = 'API_KEY_FOR_ALPHA_VANTAGE'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker'].upper()
    error, actuals, predictions, mse = predict_price(ticker)
    
    if error:
        return render_template('error.html', message=error)
    
    plot_url = plot_predictions(actuals, predictions)
    predicted_price = predictions[-1]
    
    print(f"Mean Squared Error: {mse}")
    return render_template('result.html', plot_url=plot_url, prediction=predicted_price)


@app.route('/autocomplete', methods=['GET'])


def autocomplete():
 query = request.args.get('query')
 url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={query}&apikey={ALPHA_VANTAGE_API_KEY}'
 response = requests.get(url)
 
 data = response.json()
 
 print(f"API Response: {data}")
 
 results = []
 
 if 'bestMatches' in data:
    for match in data['bestMatches']:
        results.append(f"{match['2. name']} ({match['1. symbol']})")
 return jsonify(results)

if __name__ == '__main__':
 app.run(debug=True)
