import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreprocessor:
    """数据预处理类，负责数据清洗、转换和标准化"""
    
    def __init__(self):
        """初始化数据预处理器"""
        # 定义数值型和类别型特征的预处理管道
        self.transaction_preprocessor = self._create_transaction_preprocessor()
        self.behavior_preprocessor = self._create_behavior_preprocessor()
        
    def _create_transaction_preprocessor(self):
        """创建交易数据的预处理管道"""
        numeric_features = ['amount']
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_features = ['location', 'merchant', 'payment_method']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return preprocessor
        
    def _create_behavior_preprocessor(self):
        """创建用户行为数据的预处理管道"""
        categorical_features = ['behavior_type', 'device', 'browser']
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return preprocessor
    
    def clean_transaction_data(self, transaction_data):
        """清洗交易数据
        
        参数:
            transaction_data: 原始交易数据DataFrame
            
        返回:
            清洗后的交易数据DataFrame
        """
        # 去除重复记录
        cleaned = transaction_data.drop_duplicates()
        
        # 处理缺失值
        cleaned['amount'] = cleaned['amount'].fillna(cleaned['amount'].median())
        cleaned['location'] = cleaned['location'].fillna('unknown')
        cleaned['merchant'] = cleaned['merchant'].fillna('unknown')
        cleaned['payment_method'] = cleaned['payment_method'].fillna('unknown')
        
        # 过滤异常值 - 交易金额为负或过大的值
        amount_mean = cleaned['amount'].mean()
        amount_std = cleaned['amount'].std()
        cleaned = cleaned[(cleaned['amount'] >= 0) & 
                         (cleaned['amount'] <= amount_mean + 5 * amount_std)]
                         
        # 确保时间顺序正确
        cleaned = cleaned.sort_values(['user_id', 'transaction_time'])
        
        return cleaned
    
    def clean_user_behavior_data(self, behavior_data):
        """清洗用户行为数据
        
        参数:
            behavior_data: 原始用户行为数据DataFrame
            
        返回:
            清洗后的用户行为数据DataFrame
        """
        # 去除重复记录
        cleaned = behavior_data.drop_duplicates()
        
        # 处理缺失值
        cleaned['behavior_type'] = cleaned['behavior_type'].fillna('unknown')
        cleaned['device'] = cleaned['device'].fillna('unknown')
        cleaned['browser'] = cleaned['browser'].fillna('unknown')
        cleaned['ip_address'] = cleaned['ip_address'].fillna('unknown')
        
        # 确保时间顺序正确
        cleaned = cleaned.sort_values(['user_id', 'behavior_time'])
        
        return cleaned
    
    def preprocess_transaction_features(self, transaction_data):
        """预处理交易特征
        
        参数:
            transaction_data: 清洗后的交易数据
            
        返回:
            预处理后的交易特征数组
        """
        # 选择需要的特征列
        features = transaction_data[['amount', 'location', 'merchant', 'payment_method']]
        
        # 应用预处理管道
        processed_features = self.transaction_preprocessor.fit_transform(features)
        
        return processed_features
    
    def preprocess_behavior_features(self, behavior_data):
        """预处理用户行为特征
        
        参数:
            behavior_data: 清洗后的用户行为数据
            
        返回:
            预处理后的用户行为特征数组
        """
        # 选择需要的特征列
        features = behavior_data[['behavior_type', 'device', 'browser']]
        
        # 应用预处理管道
        processed_features = self.behavior_preprocessor.fit_transform(features)
        
        return processed_features
        
    def combine_user_data(self, transaction_data, behavior_data):
        """将同一用户的交易数据和行为数据按时间对齐
        
        参数:
            transaction_data: 交易数据
            behavior_data: 用户行为数据
            
        返回:
            按用户ID分组的合并数据字典
        """
        # 获取所有唯一用户ID
        user_ids = set(transaction_data['user_id'].unique()).union(
                   set(behavior_data['user_id'].unique()))
        
        combined_data = {}
        
        for user_id in user_ids:
            # 获取该用户的交易数据
            user_transactions = transaction_data[transaction_data['user_id'] == user_id] \
                               .sort_values('transaction_time')
            
            # 获取该用户的行为数据
            user_behavior = behavior_data[behavior_data['user_id'] == user_id] \
                           .sort_values('behavior_time')
            
            combined_data[user_id] = {
                'transactions': user_transactions,
                'behavior': user_behavior
            }
            
        return combined_data
