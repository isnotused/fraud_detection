import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FraudRiskCalculator:
    """欺诈风险计算器，用于根据异常特征生成动态欺诈风险值"""
    
    def __init__(self, core_weight=0.6, trajectory_weight=0.4):
        """初始化欺诈风险计算器
        
        参数:
            core_weight: 核心异常维度方差的权重
            trajectory_weight: 用户行为轨迹变化量的权重
        """
        self.core_weight = core_weight
        self.trajectory_weight = trajectory_weight
        self.scaler = MinMaxScaler()
        
    def find_core_anomaly_dimensions(self, window_anomaly_features):
        """找出所有时间窗口的异常特征维度的交集作为核心异常维度
        
        参数:
            window_anomaly_features: 每个窗口的异常特征维度列表组成的列表
            
        返回:
            核心异常维度集合
        """
        if not window_anomaly_features or len(window_anomaly_features) == 0:
            return set()
            
        # 取所有窗口异常特征的交集
        core_dimensions = set(window_anomaly_features[0])
        for features in window_anomaly_features[1:]:
            core_dimensions.intersection_update(set(features))
            
        return core_dimensions
        
    def calculate_core_dimension_variance(self, window_data, core_dimensions):
        """计算核心异常维度的数值分布方差
        
        参数:
            window_data: 时间窗口数据
            core_dimensions: 核心异常维度集合
            
        返回:
            核心异常维度的方差均值
        """
        if not core_dimensions or 'features' not in window_data:
            return 0.0
            
        # 提取核心异常维度的特征数据
        core_features = []
        for dim in core_dimensions:
            if dim < window_data['features'].shape[1]:
                core_features.append(window_data['features'][:, dim])
                
        if not core_features:
            return 0.0
            
        # 计算每个核心维度的方差并取均值
        variances = [np.var(feat) for feat in core_features]
        return np.mean(variances)
        
    def calculate_trajectory_change(self, prev_window_data, curr_window_data):
        """计算用户行为轨迹变化量
        
        参数:
            prev_window_data: 前一时间窗口数据
            curr_window_data: 当前时间窗口数据
            
        返回:
            用户行为轨迹变化量
        """
        if prev_window_data is None or curr_window_data is None:
            return 0.0
            
        # 提取行为特征
        prev_behavior = prev_window_data.get('behavior_features', np.array([]))
        curr_behavior = curr_window_data.get('behavior_features', np.array([]))
        
        if len(prev_behavior) == 0 or len(curr_behavior) == 0:
            return 0.0
            
        # 确保特征长度相同
        min_length = min(len(prev_behavior), len(curr_behavior))
        prev_behavior = prev_behavior[:min_length]
        curr_behavior = curr_behavior[:min_length]
        
        # 计算轨迹变化量（欧氏距离）
        return np.linalg.norm(curr_behavior - prev_behavior)
        
    def calculate_risk_factor(self, window_data, prev_window_data, core_dimensions):
        """计算当前时间窗口的欺诈风险因子
        
        参数:
            window_data: 当前时间窗口数据
            prev_window_data: 前一时间窗口数据
            core_dimensions: 核心异常维度集合
            
        返回:
            当前窗口的欺诈风险因子
        """
        # 计算核心异常维度数值分布方差
        core_variance = self.calculate_core_dimension_variance(window_data, core_dimensions)
        
        # 计算用户行为轨迹变化量
        trajectory_change = self.calculate_trajectory_change(prev_window_data, window_data)
        
        # 加权组合得到风险因子
        risk_factor = (self.core_weight * core_variance) + \
                     (self.trajectory_weight * trajectory_change)
                     
        return risk_factor
        
    def calculate_dynamic_risk_values(self, all_window_data, window_anomaly_features):
        """计算所有时间窗口的动态欺诈风险值
        
        参数:
            all_window_data: 所有时间窗口的数据列表
            window_anomaly_features: 每个窗口的异常特征维度列表
            
        返回:
            每个时间窗口的动态欺诈风险值列表
        """
        if not all_window_data or len(all_window_data) == 0:
            return []
            
        # 找出核心异常维度
        core_dimensions = self.find_core_anomaly_dimensions(window_anomaly_features)
        
        # 计算每个窗口的风险因子
        risk_factors = []
        for i, window_data in enumerate(all_window_data):
            prev_window_data = all_window_data[i-1] if i > 0 else None
            risk_factor = self.calculate_risk_factor(window_data, prev_window_data, core_dimensions)
            risk_factors.append(risk_factor)
            
        # 归一化风险因子得到动态风险值
        if len(risk_factors) == 0:
            return []
            
        # 处理所有风险因子都为0的情况
        if np.max(risk_factors) == 0:
            return [0.0] * len(risk_factors)
            
        # 归一化到[0, 1]范围
        normalized_risks = self.scaler.fit_transform(np.array(risk_factors).reshape(-1, 1)).flatten()
        
        return normalized_risks.tolist()
