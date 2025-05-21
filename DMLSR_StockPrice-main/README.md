# Stock Price Prediction Using DMLSR

## 📌 Overview
Stock price prediction is a crucial area of research in financial markets, helping investors make informed decisions. This project introduces an advanced machine learning approach using **Discriminative Marginalized Least-Squares Regression (DMLSR)** to predict stock prices accurately while mitigating overfitting.

## 🚀 Key Features
- **Machine Learning-based Prediction**: Uses DMLSR for robust forecasting.
- **Real-time Data Integration**: Fetches stock data via `yFinance`.
- **Interactive UI**: Simple input for stock tickers with instant predictions.
- **Graphical Insights**: Visual representation of stock trends.
- **Overfitting Prevention**: Enhances class separability using an intraclass compactness graph.

## 🎯 Problem & Solution
Traditional models suffer from overfitting and lack of scalability. Our **DMLSR-based model** introduces a **data-reconstruction constraint** and **class separability enhancement**, improving prediction performance.

## 🏗️ Project Workflow
1. **User Input** → Enter stock ticker.
2. **Data Fetching** → Retrieve stock data via `yFinance`.
3. **Preprocessing** → Clean and format historical data.
4. **Model Training** → Train DMLSR model.
5. **Prediction** → Generate stock price forecasts.
6. **Visualization** → Display results and trends.

## 📊 Technologies Used
- **Python** (Scikit-Learn, NumPy, Pandas, Matplotlib)
- **Flask** (Web Framework)
- **Visual Studio Code** (IDE)

## 📈 Performance Results
- **Accuracy**: The model achieved an accuracy of **88.03%** in predicting stock prices, outperforming conventional regression techniques.
- **Overfitting Reduction**: The model effectively mitigates overfitting by incorporating intraclass compactness constraints, leading to improved generalization.
- **Execution Time**: Optimized for real-time stock price forecasting, with predictions generated in **less than 2 seconds** per query.
- **Comparison with Other Models**:
  - **Support Vector Machines (SVM)**: 86.03% accuracy
  - **Random Forest (RF)**: 82.65% accuracy
  - **Traditional Least-Squares Regression (LSR)**: 80.47% accuracy

## 📌 Installation & Usage
```sh
# Clone the repository
git clone https://github.com/Myth20049/DMLSR_StockPrice.git
cd stock-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🤝 Contributors
- Addanki Sai Sujith
- Mallisetti Rohan Siddarth
- Mangadoddi Karthikeya
- Sriram Rishmith Miriyala

---
_Enhancing stock market predictions with machine learning! 📈_

