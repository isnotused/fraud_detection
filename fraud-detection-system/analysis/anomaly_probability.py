import numpy as np
import pandas as pd
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler

class AnomalyProbabilityCalculator:
    """异常概率计算器，用于确定每个时间窗口在不同特征维度的异常概率"""
    
    def __init__(self, anomaly_threshold=0.7):
        """初始化异常概率计算器
        
        参数:
            anomaly_threshold: 异常特征维度的判定阈值
        """
        self.anomaly_threshold = anomaly_threshold
        self.scaler = MinMaxScaler()
        
    def calculate_frequency_energy(self, feature_sequence):
        """计算特征序列在频域的能量
        
        参数:
            feature_sequence: 特征序列
            
        返回:
            频域能量值
        """
        # 计算傅里叶变换
        fft_vals = fft(feature_sequence)
        
        # 计算能量（频谱的平方和）
        energy = np.sum(np.abs(fft_vals) **2)
        
        return energy
        
    def calculate_energy_difference(self, prev_sequence, curr_sequence):
        """计算连续两个时间窗口特征在频域的能量差异
        
        参数:
            prev_sequence: 前一时间窗口的特征序列
            curr_sequence: 当前时间窗口的特征序列
            
        返回:
            能量差异值
        """
        # 确保序列长度相同
        min_length = min(len(prev_sequence), len(curr_sequence))
        prev_sequence = prev_sequence[:min_length]
        curr_sequence = curr_sequence[:min_length]
        
        # 计算各自的频域能量
        prev_energy = self.calculate_frequency_energy(prev_sequence)
        curr_energy = self.calculate_frequency_energy(curr_sequence)
        
        # 计算相对能量差异
        if prev_energy == 0 and curr_energy == 0:
            return 0.0
        elif prev_energy == 0:
            return 1.0
            
        return abs(curr_energy - prev_energy) / prev_energy
        
    def calculate_value_change_rate(self, prev_values, curr_values):
        """计算数值变化率
        
        参数:
            prev_values: 前一时间窗口的特征值
            curr_values: 当前时间窗口的特征值
            
        返回:
            数值变化率
        """
        # 确保输入数组长度相同
        min_length = min(len(prev_values), len(curr_values))
        prev_values = prev_values[:min_length]
        curr_values = curr_values[:min_length]
        
        # 避免除零错误
        non_zero_mask = prev_values != 0
        if np.sum(non_zero_mask) == 0:
            return 0.0
            
        # 计算变化率
        changes = np.abs(curr_values[non_zero_mask] - prev_values[non_zero_mask]) / \
                  np.abs(prev_values[non_zero_mask])
                  
        return np.mean(changes)
        
    def calculate_distribution_shift(self, prev_dist, curr_dist):
        """计算数据分布偏移度
        
        参数:
            prev_dist: 前一时间窗口的特征分布
            curr_dist: 当前时间窗口的特征分布
            
        返回:
            分布偏移度，范围[0, 1]
        """
        # 确保分布长度相同
        min_length = min(len(prev_dist), len(curr_dist))
        prev_dist = prev_dist[:min_length]
        curr_dist = curr_dist[:min_length]
        
        # 标准化分布
        prev_norm = prev_dist / np.sum(prev_dist) if np.sum(prev_dist) != 0 else prev_dist
        curr_norm = curr_dist / np.sum(curr_dist) if np.sum(curr_dist) != 0 else curr_dist
        
        # 计算KL散度（添加微小值避免log(0)）
        eps = 1e-10
        kl_div = np.sum(prev_norm * np.log((prev_norm + eps) / (curr_norm + eps)))
        
        # 计算JS散度（对称版本的KL散度）
        m = 0.5 * (prev_norm + curr_norm)
        js_div = 0.5 * np.sum(prev_norm * np.log((prev_norm + eps) / (m + eps))) + \
                 0.5 * np.sum(curr_norm * np.log((curr_norm + eps) / (m + eps)))
                 
        # JS散度范围是[0, log(2)]，归一化到[0, 1]
        return js_div / np.log(2) if np.log(2) != 0 else 0.0
        
    def calculate_temporary_anomaly_coefficient(self, prev_sequence, curr_sequence, 
                                              prev_dist, curr_dist, consistency_change):
        """计算临时异常系数
        
        参数:
            prev_sequence: 前一时间窗口的特征序列
            curr_sequence: 当前时间窗口的特征序列
            prev_dist: 前一时间窗口的特征分布
            curr_dist: 当前时间窗口的特征分布
            consistency_change: 行为一致性指标变化幅度
            
        返回:
            临时异常系数
        """
        # 计算数值变化率
        value_rate = self.calculate_value_change_rate(prev_sequence, curr_sequence)
        
        # 计算分布偏移度
        dist_shift = self.calculate_distribution_shift(prev_dist, curr_dist)
        
        # 临时异常系数由数值变化率、分布偏移度和一致性变化共同决定
        # 三者权重相等
        temp_coeff = (value_rate + dist_shift + consistency_change) / 3
        
        return temp_coeff
        
    def calculate_feature_anomaly_probability(self, prev_window_data, curr_window_data, 
                                            consistency_score, consistency_change, 
                                            feature_dim):
        """计算特定特征维度的异常概率
        
        参数:
            prev_window_data: 前一时间窗口的数据
            curr_window_data: 当前时间窗口的数据
            consistency_score: 当前窗口的行为一致性指标
            consistency_change: 行为一致性指标变化幅度
            feature_dim: 特征维度索引
            
        返回:
            该特征维度的异常概率
        """
        # 提取特征序列和分布
        prev_sequence = prev_window_data['features'][:, feature_dim]
        curr_sequence = curr_window_data['features'][:, feature_dim]
        
        prev_dist = prev_window_data['distributions'][feature_dim]
        curr_dist = curr_window_data['distributions'][feature_dim]
        
        # 计算能量差异
        energy_diff = self.calculate_energy_difference(prev_sequence, curr_sequence)
        
        # 计算临时异常系数
        temp_coeff = self.calculate_temporary_anomaly_coefficient(
            prev_sequence, curr_sequence, 
            prev_dist, curr_dist, 
            consistency_change
        )
        
        # 计算一致性与临时异常系数的比值
        if temp_coeff == 0:
            ratio = 1.0 if consistency_score > 0 else 0.0
        else:
            ratio = consistency_score / temp_coeff
            
        # 归一化比值
        normalized_ratio = self.scaler.fit_transform([[ratio]])[0][0]
        
        # 预设基准值为1.0，计算异常概率
        # 异常概率 = 基准值 - 归一化比值 + 能量差异的权重部分
        anomaly_prob = 1.0 - normalized_ratio + 0.3 * energy_diff
        
        # 确保异常概率在[0, 1]范围内
        return max(0.0, min(1.0, anomaly_prob))
        
    def calculate_window_anomaly_probabilities(self, prev_window_data, curr_window_data, 
                                             consistency_score, consistency_change):
        """计算当前时间窗口所有特征维度的异常概率
        
        参数:
            prev_window_data: 前一时间窗口的数据
            curr_window_data: 当前时间窗口的数据
            consistency_score: 当前窗口的行为一致性指标
            consistency_change: 行为一致性指标变化幅度
            
        返回:
            每个特征维度的异常概率列表
        """
        if prev_window_data is None or curr_window_data is None:
            num_features = curr_window_data['features'].shape[1] if curr_window_data else \
                          prev_window_data['features'].shape[1] if prev_window_data else 0
            return [0.0] * num_features
            
        num_features = curr_window_data['features'].shape[1]
        anomaly_probs = []
        
        for dim in range(num_features):
            prob = self.calculate_feature_anomaly_probability(
                prev_window_data, curr_window_data,
                consistency_score, consistency_change,
                dim
            )
            anomaly_probs.append(prob)
            
        return anomaly_probs
        
    def identify_anomaly_features(self, anomaly_probabilities):
        """识别异常特征维度
        
        参数:
            anomaly_probabilities: 各特征维度的异常概率列表
            
        返回:
            异常特征维度的索引列表
        """
        return [i for i, prob in enumerate(anomaly_probabilities) 
                if prob > self.anomaly_threshold]
