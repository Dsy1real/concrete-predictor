# NN_numpy.py
import json
import numpy as np
import os
import sys  # 导入 sys 模块


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))

    return os.path.join(base_path, relative_path)
SCALER_PATH = resource_path('scaler.npz')
WEIGHTS_PATH = resource_path('model_weights.json')


class NumpyNeuralNetwork:
    """一个使用 NumPy 实现的简单神经网络，用于预测。"""
    def __init__(self):
        self.weights = None
        self.biases = None
        self.scaler_mean = None
        self.scaler_std = None
        self._load_model()

    def _load_model(self):
        try:
            with open(WEIGHTS_PATH, 'r') as f:
                model_data = json.load(f)
            all_keys = sorted(model_data.keys())
            weights_list = [model_data[k] for k in all_keys if k.endswith('.weight')]
            biases_list = [model_data[k] for k in all_keys if k.endswith('.bias')]
            if not weights_list or not biases_list:
                raise KeyError("JSON文件中未能找到权重或偏置数据。")
            self.weights = [np.array(w).T for w in weights_list]
            self.biases = [np.array(b) for b in biases_list]
        except FileNotFoundError:
            print(f"错误: 找不到模型权重文件 '{WEIGHTS_PATH}'。")
            raise
        except json.JSONDecodeError:
            print(f"错误: '{WEIGHTS_PATH}' 不是一个有效的JSON文件。")
            raise

        try:
            scaler_data = np.load(SCALER_PATH)
            self.scaler_mean = scaler_data['mean']
            self.scaler_std = scaler_data['std']
        except FileNotFoundError:
            print(f"错误: 找不到标准化参数文件 '{SCALER_PATH}'。")
            raise

    def _relu(self, x):
        return np.maximum(0, x)
    def predict(self, X):
        if self.weights is None or self.scaler_mean is None:
            raise RuntimeError("模型或标准化参数未加载，无法进行预测。")
        X_scaled = (X - self.scaler_mean) / self.scaler_std
        layer_output = X_scaled
        for i in range(len(self.weights) - 1):
            layer_output = self._relu(np.dot(layer_output, self.weights[i]) + self.biases[i])
        final_output = np.dot(layer_output, self.weights[-1]) + self.biases[-1]
        return final_output.flatten()


model = NumpyNeuralNetwork()


def data_test(input_csv_path):
    try:
        test_data = np.loadtxt(input_csv_path, delimiter=',', skiprows=1)
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]
    except FileNotFoundError:
        print(f"错误: 测试文件 '{input_csv_path}' 未找到。")
        raise

    predictions = model.predict(X_test)
    mse = np.mean((predictions - y_test) ** 2)
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {
        "MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2,
        "predictions": predictions, "true_values": y_test
    }


def predict_single(features):
    if len(features) != 8:
        raise ValueError(f"输入需要8个特征值，但收到了 {len(features)} 个。")
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]
