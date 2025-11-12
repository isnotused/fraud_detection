import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
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
        if historical_params is None:
            historical_params = self.params.copy()

        # 如果风险值为空，直接返回历史参数
        if not dynamic_risk_values:
            return historical_params

        temp_params = {}
        window_ids = sorted(dynamic_risk_values.keys())

        for i, window_id in enumerate(window_ids):
            risk_value = dynamic_risk_values.get(window_id, 0)
            # 处理 NaN 风险值
            if risk_value is None or (isinstance(risk_value, float) and np.isnan(risk_value)):
                risk_value = 0
            window_params = {}
            for param, value in historical_params.items():
                if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    base = value if isinstance(value, (int, float)) and not np.isnan(value) else 1
                    scaled = base * (1 + risk_value)
                    # 防止 NaN 或非整数进入 int()
                    if isinstance(scaled, float) and np.isnan(scaled):
                        scaled = base
                    window_params[param] = max(1, int(round(scaled)))
                elif isinstance(value, (int, float)):
                    scaled = value * (1 + risk_value)
                    if isinstance(scaled, float) and np.isnan(scaled):
                        scaled = value
                    window_params[param] = scaled
                else:
                    window_params[param] = value

            temp_params[window_id] = window_params

            if i % 20 == 0:
                time.sleep(0.05)

        final_params = {}
        for param in historical_params.keys():
            if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                param_values = [p[param] for p in temp_params.values() if isinstance(p[param], (int, float)) and not np.isnan(p[param])]
                mean_val = np.mean(param_values) if param_values else historical_params[param]
                if isinstance(mean_val, float) and np.isnan(mean_val):
                    mean_val = historical_params[param]
                final_params[param] = max(1, int(round(mean_val)))
            elif isinstance(historical_params[param], (int, float)):
                param_values = [p[param] for p in temp_params.values() if isinstance(p[param], (int, float)) and not np.isnan(p[param])]
                mean_val = np.mean(param_values) if param_values else historical_params[param]
                if isinstance(mean_val, float) and np.isnan(mean_val):
                    mean_val = historical_params[param]
                final_params[param] = mean_val
            else:
                final_params[param] = historical_params[param]

        self.params = final_params
        self.model.set_params(**final_params)
        return final_params
    
    def train(self, X, y):
        """训练模型"""
        # 空数据防护
        if X is None or y is None or len(X) == 0:
            raise ValueError("训练数据为空，无法训练模型")

        # 检查标签类别数量
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            # 使用 DummyClassifier 处理单一类别场景，避免 RandomForest 抛出异常
            print(f"\n警告：训练数据仅包含 {len(self.classes_)} 个类别，将使用 DummyClassifier 作为占位模型")
            dummy = DummyClassifier(strategy="constant", constant=self.classes_[0])
            dummy.fit(np.zeros((len(X), 1)), y)  # 训练一个简单的占位模型
            self.model = dummy
            self.is_trained = True
            # 单一类别的 '准确率' 视为 1.0（训练集全预测正确）
            return 1.0

        # 正常训练流程
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        total_steps = 10
        for _ in range(total_steps):
            self.model.fit(X_train, y_train)
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