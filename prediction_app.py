from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the dataset
#df = pd.read_csv('/Users/kranthikuamarchowdary/Desktop/stock-prediction-dashboard/dataset/sentiment_annotated_with_texts.csv')

df = pd.read_csv('/Users/kranthikuamarchowdary/Desktop/stock-prediction-dashboard/dataset/sentiment_annotated_with_texts.csv')
df['published_at'] = pd.to_datetime(df['published_at'])

@app.route('/')
def index():
    stocks = df['ticker'].unique()  # Get unique stock tickers
    return render_template('index.html', stocks=stocks)

@app.route('/predict', methods=['POST'])
def predict():
    selected_stock = request.form.get('stock')
    stock_data = df[df['ticker'] == selected_stock]  # Filter data by selected stock
    
    plt.figure(figsize=(10, 5))
    plt.plot(stock_data['published_at'], stock_data['finbert_sent_score'], label='Sentiment Score', linestyle='--')  # Adjust column names if needed
    plt.plot(stock_data['published_at'], stock_data['true_sentiment'], label='Stock Price')  # Adjust column names if needed
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title(f'{selected_stock}: Price & Sentiment Trend')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/stock_plot.png')
    plt.close()

    return render_template('index.html', stock_image='stock_plot.png', stocks=df['ticker'].unique())  # Update column names if needed

if __name__ == '__main__':
    app.run(port=5002, debug=True)  # Run on a different port
