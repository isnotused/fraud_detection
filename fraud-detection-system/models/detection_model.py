import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

class FraudDetectionModel(BaseEstimator, ClassifierMixin):
    """欺诈检测模型基类"""
    
    def __init__(self, model_type='rf', random_state=42):
        """初始化欺诈检测模型
        
        参数:
            model_type: 模型类型，'rf'表示随机森林，'gb'表示梯度提升树
            random_state: 随机种子，保证结果可复现
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.is_trained = False
        
    def _create_model(self):
        """创建基础模型"""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
    def fit(self, X, y):
        """训练模型
        
        参数:
            X: 特征数据
            y: 标签数据（1表示欺诈，0表示正常）
        """
        self.model.fit(X, y)
        self.is_trained = True
        return self
        
    def predict(self, X):
        """预测样本是否为欺诈
        
        参数:
            X: 特征数据
            
        返回:
            预测结果，1表示欺诈，0表示正常
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法训练模型")
        return self.model.predict(X)
        
    def predict_proba(self, X):
        """预测样本为欺诈的概率
        
        参数:
            X: 特征数据
            
        返回:
            欺诈概率
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法训练模型")
        return self.model.predict_proba(X)[:, 1]
        
    def evaluate(self, X, y):
        """评估模型性能
        
        参数:
            X: 特征数据
            y: 标签数据
            
        返回:
            包含各项评估指标的字典
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法训练模型")
            
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        return {
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': roc_auc_score(y, y_prob)
        }
        
    def cross_validate(self, X, y, cv=5):
        """交叉验证评估模型
        
        参数:
            X: 特征数据
            y: 标签数据
            cv: 交叉验证折数
            
        返回:
            包含各项评估指标交叉验证结果的字典
        """
        scoring = ['precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'precision': cv_results['test_precision'].mean(),
            'recall': cv_results['test_recall'].mean(),
            'f1': cv_results['test_f1'].mean(),
            'auc': cv_results['test_roc_auc'].mean()
        }
        
    def get_feature_importance(self):
        """获取特征重要性
        
        返回:
            特征重要性数组
        """
        if not self.is_trained:
            raise RuntimeError("模型尚未训练，请先调用fit方法训练模型")
            
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            raise NotImplementedError("该模型不支持特征重要性计算")
            
    def save_model(self, file_path):
        """保存模型到文件
        
        参数:
            file_path: 保存路径
        """
        import joblib
        joblib.dump(self, file_path)
        
    @classmethod
    def load_model(cls, file_path):
        """从文件加载模型
        
        参数:
            file_path: 模型文件路径
            
        返回:
            加载的模型
        """
        import joblib
        return joblib.load(file_path)
