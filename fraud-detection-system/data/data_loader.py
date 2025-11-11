import pandas as pd
import numpy as np
import os

class DataLoader:
    """数据加载器，负责读取金融交易数据和用户行为数据"""
    
    def __init__(self, transaction_data_path, user_behavior_data_path):
        """初始化数据加载器
        
        参数:
            transaction_data_path: 交易数据文件路径
            user_behavior_data_path: 用户行为数据文件路径
        """
        self.transaction_path = transaction_data_path
        self.user_behavior_path = user_behavior_data_path
        
    def load_transaction_data(self):
        """加载金融消费交易数据
        
        返回:
            包含交易数据的DataFrame
        """
        if not os.path.exists(self.transaction_path):
            raise FileNotFoundError(f"交易数据文件不存在: {self.transaction_path}")
            
        # 根据文件扩展名选择合适的读取方法
        if self.transaction_path.endswith('.csv'):
            return pd.read_csv(self.transaction_path, parse_dates=['transaction_time'])
        elif self.transaction_path.endswith('.parquet'):
            return pd.read_parquet(self.transaction_path)
        elif self.transaction_path.endswith('.json'):
            return pd.read_json(self.transaction_path, convert_dates=['transaction_time'])
        else:
            raise ValueError("不支持的交易数据文件格式")
            
    def load_user_behavior_data(self):
        """加载多维用户行为数据
        
        返回:
            包含用户行为数据的DataFrame
        """
        if not os.path.exists(self.user_behavior_path):
            raise FileNotFoundError(f"用户行为数据文件不存在: {self.user_behavior_path}")
            
        # 根据文件扩展名选择合适的读取方法
        if self.user_behavior_path.endswith('.csv'):
            return pd.read_csv(self.user_behavior_path, parse_dates=['behavior_time'])
        elif self.user_behavior_path.endswith('.parquet'):
            return pd.read_parquet(self.user_behavior_path)
        elif self.user_behavior_path.endswith('.json'):
            return pd.read_json(self.user_behavior_path, convert_dates=['behavior_time'])
        else:
            raise ValueError("不支持的用户行为数据文件格式")
            
    def load_all_data(self):
        """加载所有需要的数据
        
        返回:
            交易数据和用户行为数据的元组
        """
        transaction_data = self.load_transaction_data()
        user_behavior_data = self.load_user_behavior_data()
        
        # 确保数据中包含必要的字段
        self._validate_transaction_data(transaction_data)
        self._validate_user_behavior_data(user_behavior_data)
        
        return transaction_data, user_behavior_data
        
    def _validate_transaction_data(self, data):
        """验证交易数据是否包含必要的字段"""
        required_fields = ['user_id', 'transaction_time', 'amount', 'location', 'merchant', 'payment_method']
        for field in required_fields:
            if field not in data.columns:
                raise ValueError(f"交易数据缺少必要字段: {field}")
                
    def _validate_user_behavior_data(self, data):
        """验证用户行为数据是否包含必要的字段"""
        required_fields = ['user_id', 'behavior_time', 'behavior_type', 'device', 'ip_address', 'browser']
        for field in required_fields:
            if field not in data.columns:
                raise ValueError(f"用户行为数据缺少必要字段: {field}")
