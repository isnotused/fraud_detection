import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class FeatureExtractor:
    """特征提取器，负责从交易数据和用户行为数据中提取特征"""
    
    def __init__(self):
        """初始化特征提取器"""
        self.tfidf_vectorizer = TfidfVectorizer(min_df=1)
        
    def extract_transaction_features(self, transaction_data):
        """从交易数据中提取特征
        
        参数:
            transaction_data: 交易数据DataFrame
            
        返回:
            提取的交易特征DataFrame
        """
        if transaction_data.empty:
            return pd.DataFrame()
            
        # 基础数值特征
        features = pd.DataFrame()
        features['transaction_count'] = [len(transaction_data)]
        features['total_amount'] = [transaction_data['amount'].sum()]
        features['avg_amount'] = [transaction_data['amount'].mean()]
        features['max_amount'] = [transaction_data['amount'].max()]
        features['min_amount'] = [transaction_data['amount'].min()]
        features['amount_std'] = [transaction_data['amount'].std() if len(transaction_data) > 1 else 0]
        
        # 交易频率特征
        if len(transaction_data) > 1:
            times = transaction_data['transaction_time'].sort_values()
            time_diff = (times.iloc[1:] - times.iloc[:-1]).dt.total_seconds() / 60
            features['avg_interval'] = [time_diff.mean()]
            features['interval_std'] = [time_diff.std() if len(time_diff) > 1 else 0]
        else:
            features['avg_interval'] = [0]
            features['interval_std'] = [0]
            
        # 类别型特征统计
        location_counts = transaction_data['location'].value_counts(normalize=True)
        features['location_diversity'] = [len(location_counts)]
        features['primary_location_ratio'] = [location_counts.iloc[0] if len(location_counts) > 0 else 0]
        
        merchant_counts = transaction_data['merchant'].value_counts(normalize=True)
        features['merchant_diversity'] = [len(merchant_counts)]
        features['primary_merchant_ratio'] = [merchant_counts.iloc[0] if len(merchant_counts) > 0 else 0]
        
        payment_counts = transaction_data['payment_method'].value_counts(normalize=True)
        features['payment_diversity'] = [len(payment_counts)]
        features['primary_payment_ratio'] = [payment_counts.iloc[0] if len(payment_counts) > 0 else 0]
        
        return features
        
    def extract_behavior_features(self, behavior_data):
        """从用户行为数据中提取特征
        
        参数:
            behavior_data: 用户行为数据DataFrame
            
        返回:
            提取的用户行为特征DataFrame
        """
        if behavior_data.empty:
            return pd.DataFrame()
            
        # 基础行为特征
        features = pd.DataFrame()
        features['behavior_count'] = [len(behavior_data)]
        
        # 行为频率特征
        if len(behavior_data) > 1:
            times = behavior_data['behavior_time'].sort_values()
            time_diff = (times.iloc[1:] - times.iloc[:-1]).dt.total_seconds() / 60
            features['avg_behavior_interval'] = [time_diff.mean()]
            features['behavior_interval_std'] = [time_diff.std() if len(time_diff) > 1 else 0]
        else:
            features['avg_behavior_interval'] = [0]
            features['behavior_interval_std'] = [0]
            
        # 行为类型特征
        behavior_counts = behavior_data['behavior_type'].value_counts(normalize=True)
        features['behavior_type_diversity'] = [len(behavior_counts)]
        features['primary_behavior_ratio'] = [behavior_counts.iloc[0] if len(behavior_counts) > 0 else 0]
        
        # 设备特征
        device_counts = behavior_data['device'].value_counts(normalize=True)
        features['device_diversity'] = [len(device_counts)]
        features['primary_device_ratio'] = [device_counts.iloc[0] if len(device_counts) > 0 else 0]
        
        # 浏览器特征
        browser_counts = behavior_data['browser'].value_counts(normalize=True)
        features['browser_diversity'] = [len(browser_counts)]
        features['primary_browser_ratio'] = [browser_counts.iloc[0] if len(browser_counts) > 0 else 0]
        
        # IP地址特征
        ip_counts = behavior_data['ip_address'].value_counts(normalize=True)
        features['ip_diversity'] = [len(ip_counts)]
        features['primary_ip_ratio'] = [ip_counts.iloc[0] if len(ip_counts) > 0 else 0]
        
        return features
        
    def extract_window_features(self, window_data):
        """从时间窗口数据中提取综合特征
        
        参数:
            window_data: 包含交易和行为数据的窗口数据
            
        返回:
            提取的窗口综合特征DataFrame
        """
        trans_features = self.extract_transaction_features(window_data['transactions'])
        behav_features = self.extract_behavior_features(window_data['behavior'])
        
        # 如果两个特征集都为空，返回空DataFrame
        if trans_features.empty and behav_features.empty:
            return pd.DataFrame()
            
        # 如果其中一个为空，返回另一个
        if trans_features.empty:
            return behav_features
        if behav_features.empty:
            return trans_features
            
        # 合并特征
        combined_features = pd.concat([trans_features.reset_index(drop=True), 
                                      behav_features.reset_index(drop=True)], axis=1)
        
        # 添加时间特征
        combined_features['window_duration'] = [(window_data['window_end'] - window_data['window_start']).total_seconds() / 60]
        
        # 添加交易与行为的比率特征
        if combined_features['behavior_count'].iloc[0] > 0:
            combined_features['trans_behavior_ratio'] = [
                combined_features['transaction_count'].iloc[0] / combined_features['behavior_count'].iloc[0]
            ]
        else:
            combined_features['trans_behavior_ratio'] = [0]
            
        return combined_features
        
    def extract_user_features(self, user_window_data):
        """从用户的所有时间窗口数据中提取特征序列
        
        参数:
            user_window_data: 按窗口组织的用户数据
            
        返回:
            包含所有窗口特征的字典，键为窗口索引
        """
        user_features = {}
        
        for window_idx, window_data in user_window_data.items():
            window_features = self.extract_window_features(window_data)
            if not window_features.empty:
                user_features[window_idx] = window_features
            else:
                user_features[window_idx] = None
                
        return user_features
