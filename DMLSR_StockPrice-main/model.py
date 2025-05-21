import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
import io
import base64


def fetch_data(ticker):
    try:
        data = yf.download(ticker, period='1y', interval='1d')
        if data.empty:
            return None, f"No data found for ticker: {ticker}"
        return data, None
    except Exception as e:
        return None, str(e)


def preprocess_data(data):
    window_size = 1
    chunk_size = 1
    
    num_rows = data.shape[0]
    rounded_num_rows = (num_rows // chunk_size) * chunk_size
    data = data.iloc[:rounded_num_rows]
    features_list = []
    labels_list = []
    for i in range(len(data) - window_size):
        features_list.append(data['Close'][i:i + window_size].values)
        labels_list.append(data['Close'][i + window_size])
        
    features = np.array(features_list)
    labels = np.array(labels_list)
    
    feature_scaler = StandardScaler()
    label_scaler = StandardScaler()
    
    features = feature_scaler.fit_transform(features)
    labels = label_scaler.fit_transform(labels.reshape(-1, 1)).flatten()
    
    return features, labels, feature_scaler, label_scaler


def compute_laplacian(W):
    D = np.diag(np.sum(W, axis=1))
    return D - W


def compute_W(X, sigma):
    from sklearn.metrics.pairwise import rbf_kernel
    return rbf_kernel(X, gamma=1 / (2 * sigma ** 2))


def compute_G1(X, L):
    return X @ L @ X.T


def compute_G2(X, gamma, mu):
    return X + (gamma / mu)


def solve_l21_regularization(A, lambd):
    E = np.zeros_like(A)
    for i in range(A.shape[1]):
        norm = np.linalg.norm(A[:, i], 2)
        if norm > lambd:
            E[:, i] = (1 - lambd / norm) * A[:, i]
    return E


def DMLSR(X, L, P, lambda1, lambda2, lambda3, mu, gamma, sigma, rho, mu_max, theta,
          iterations):
    G1 = compute_G1(X, L)
    G2 = compute_G2(X, gamma, mu)
    R = np.zeros((X.shape[1], X.shape[0]))
    for _ in range(iterations):
        Q = np.linalg.inv(lambda1 * np.eye(X.shape[0]) + (1 + mu) * (X @ X.T) + lambda2 *
                        G1) @ (mu * X @ (G2.T @ P).T + X @ R)
        E = solve_l21_regularization(X.T @ Q - X, lambda3).T
        R = (X - Q.T @ X).T
        P = np.linalg.qr(X.T @ Q - G2 @ R + E.T)[0]
        mu = min(rho * mu, mu_max)
        gamma = gamma + mu * (X - P @ Q.T @ X)
    return Q, R, P


def predict_price(ticker):
    data, error = fetch_data(ticker)
    if error:
        return error, None, None, None
    features, labels, feature_scaler, label_scaler = preprocess_data(data)
    lambda1, lambda2, lambda3 = 0.5, 0.5, 0.5
    mu, gamma, sigma = 1e-8, 0, 0.5
    rho, mu_max = 1.5, 1
    theta = 1
    iterations = 100
    P = np.eye(1)
    predictions = []
    actuals = []
    for start_idx in range(0, len(features), 1):
        end_idx = start_idx + 1
        if end_idx > len(features):
            break
        X_train = features[start_idx:end_idx]
        y_train = labels[start_idx:end_idx]
        W = compute_W(X_train, sigma)
        L = compute_laplacian(W)
        Q, R, P = DMLSR(X_train.T, L, P, lambda1, lambda2, lambda3, mu, gamma, sigma,
                        rho, mu_max, theta, iterations)
        y_pred_train = (Q.T @ X_train.T).mean(axis=0)
        y_pred_train = label_scaler.inverse_transform(
            y_pred_train.reshape(-1, 1)).flatten()
        y_train = label_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        predictions.extend(y_pred_train)
        actuals.extend(y_train)
    mse_train = mean_squared_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"Mean Squared Error: {mse_train}")
    print(f"R^2 Score: {r2}")
    
    return None, actuals, predictions, mse_train


def plot_predictions(actuals, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(actuals, label='Actual', color='#B1B6A6')
    plt.plot(predictions, label='Predicted', color='#39FF14')
    plt.legend()
    plt.title('Stock Price Predictions', color='#B1B6A6')
    plt.xlabel('Days', color='#B1B6A6')
    plt.ylabel('Price', color='#B1B6A6')
    plt.grid(True, color='#696773')
    plt.gca().set_facecolor('#000000')
    plt.gca().spines['bottom'].set_color('#B1B6A6')
    plt.gca().spines['top'].set_color('#B1B6A6')
    plt.gca().spines['left'].set_color('#B1B6A6')
    plt.gca().spines['right'].set_color('#B1B6A6')
    plt.gca().tick_params(axis='x', colors='#B1B6A6')
    plt.gca().tick_params(axis='y', colors='#B1B6A6')
    img = io.BytesIO()
    plt.savefig(img, format='png', facecolor='#000000')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return plot_url
