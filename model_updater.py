import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from progress_bar import update_progress
import time

class FraudDetectionModel:
    """欺诈检测模型类"""
    
    def __init__(self):
        # 初始化模型参数
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42
        )
        self.params = {
            'n_estimators': 50,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'class_weight': 'balanced'
        }
        self.is_trained = False
        self.classes_ = None  # 存储训练数据的类别
    
    def update_parameters(self, dynamic_risk_values, historical_params=None):
        """基于动态欺诈风险值更新检测模型参数"""
        progress = 0
        update_progress(progress, "更新模型参数")
        
        if historical_params is None:
            historical_params = self.params.copy()
        
        temp_params = {}
        window_ids = sorted(dynamic_risk_values.keys())
        total_windows = len(window_ids)
        
        for i, window_id in enumerate(window_ids):
            risk_value = dynamic_risk_values[window_id]
            window_params = {}
            for param, value in historical_params.items():
                if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    window_params[param] = max(1, int(value * (1 + risk_value)))
                elif isinstance(value, (int, float)):
                    window_params[param] = value * (1 + risk_value)
                else:
                    window_params[param] = value
            
            temp_params[window_id] = window_params
            progress = (i + 1) / total_windows
            update_progress(progress, "更新模型参数")
            if i % 20 == 0:
                time.sleep(0.05)
        
        final_params = {}
        for param in historical_params.keys():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                param_values = [p[param] for p in temp_params.values()]
                final_params[param] = max(1, int(np.mean(param_values)))
            elif isinstance(historical_params[param], (int, float)):
                param_values = [p[param] for p in temp_params.values()]
                final_params[param] = np.mean(param_values)
            else:
                final_params[param] = historical_params[param]
        
        self.params = final_params
        self.model.set_params(**final_params)
        update_progress(1, "更新模型参数")
        return final_params
    
    def train(self, X, y):
        """训练模型"""
        progress = 0
        update_progress(progress, "训练模型")
        
        # 检查标签是否包含两个类别
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            print(f"\n警告：训练数据仅包含{len(self.classes_)}个类别，可能影响模型效果")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        total_steps = 10
        for i in range(total_steps):
            self.model.fit(X_train, y_train)
            progress = (i + 1) / total_steps
            update_progress(progress, "训练模型")
            time.sleep(0.2)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n模型训练完成，准确率: {accuracy:.4f}")
        print("分类报告:")
        print(classification_report(y_test, y_pred, zero_division=1))  # 处理无预测样本的类别
        
        self.is_trained = True
        return accuracy
    
    def detect_fraud(self, data):
        """检测欺诈行为"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，请先训练模型")
        
        # 预测概率，处理单类别情况
        fraud_prob = self.model.predict_proba(data)
        if fraud_prob.shape[1] == 1:
            fraud_probs = np.zeros(len(data))
            fraud_labels = np.zeros(len(data), dtype=int)
        else:
            fraud_probs = fraud_prob[:, 1]
            fraud_labels = (fraud_probs > 0.5).astype(int)
        
        return fraud_labels, fraud_probs