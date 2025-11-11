import numpy as np
import pandas as pd
from scipy.fftpack import fft
from progress_bar import update_progress
import time

def calculate_frequency_energy(transaction_features, feature_dim):
    """计算交易特征在频域的能量"""
    # 提取特征序列
    feature_sequence = []
    for window_id in sorted(transaction_features["window_id"].unique()):
        window_data = transaction_features[transaction_features["window_id"] == window_id]
        if not window_data.empty:
            feature_sequence.append(window_data.iloc[0][feature_dim])
        else:
            feature_sequence.append(0)
    
    # 傅里叶变换
    fft_vals = fft(feature_sequence)
    # 计算能量
    energy = np.abs(fft_vals) **2
    return energy

def calculate_user_behavior_change(behavior_distributions, window_id1, window_id2):
    """计算用户行为轨迹变化量"""
    dist1 = behavior_distributions.get(window_id1, np.array([]))
    dist2 = behavior_distributions.get(window_id2, np.array([]))
    
    # 确保分布长度一致
    min_len = min(len(dist1), len(dist2))
    if min_len == 0:
        return 0.0
    
    dist1 = dist1[:min_len]
    dist2 = dist2[:min_len]
    
    # 计算KL散度作为变化量
    eps = 1e-10
    kl_div = np.sum(dist1 * np.log((dist1 + eps) / (dist2 + eps)))
    return min(kl_div, 5)  # 限制最大值

def determine_anomaly_probability(transaction_features, behavior_distributions, consistency_scores, time_offsets):
    """确定每个时间窗口在不同特征维度的异常概率"""
    progress = 0
    update_progress(progress, "计算异常概率")
    
    # 定义特征维度
    feature_dims = ["total_transactions", "total_amount", "avg_amount", "fraud_ratio"]
    window_ids = sorted(transaction_features["window_id"].unique())
    total_windows = len(window_ids)
    anomaly_probabilities = {dim: {} for dim in feature_dims}
    
    # 预设基准值
    base_value = 1.0
    
    # 预先计算所有偏移量的绝对值
    all_offsets = [abs(offset) for offset in time_offsets.values()]
    avg_all_offset = np.mean(all_offsets) if all_offsets else 0  # 所有偏移的平均值
    
    for i, window_id in enumerate(window_ids):
        if window_id == 0:
            # 第一个窗口没有前序窗口，异常概率设为0
            for dim in feature_dims:
                anomaly_probabilities[dim][window_id] = 0.0
            progress = (i + 1) / total_windows
            update_progress(progress, "计算异常概率")
            continue
        
        prev_window_id = window_id - 1
        
        # 获取当前窗口和前一个窗口的特征
        current_window = transaction_features[transaction_features["window_id"] == window_id]
        prev_window = transaction_features[transaction_features["window_id"] == prev_window_id]
        
        if current_window.empty or prev_window.empty:
            for dim in feature_dims:
                anomaly_probabilities[dim][window_id] = 0.5  
            progress = (i + 1) / total_windows
            update_progress(progress, "计算异常概率")
            continue
        
        current_window = current_window.iloc[0]
        prev_window = prev_window.iloc[0]
        
        # 计算频域能量差异
        energy_diffs = {}
        for dim in feature_dims:
            # 为当前和前序窗口构建特征序列
            prev_energy = calculate_frequency_energy(transaction_features[transaction_features["window_id"] <= prev_window_id], dim)
            curr_energy = calculate_frequency_energy(transaction_features[transaction_features["window_id"] <= window_id], dim)
            
            # 确保能量数组长度一致
            min_len = min(len(prev_energy), len(curr_energy))
            energy_diff = np.sum(np.abs(prev_energy[:min_len] - curr_energy[:min_len])) / min_len
            energy_diffs[dim] = min(energy_diff, 1.0)  # 归一化
        
        # 计算用户行为轨迹变化量
        behavior_change = calculate_user_behavior_change(behavior_distributions, prev_window_id, window_id)
        
        # 计算每个特征维度的异常概率
        for dim in feature_dims:
            # 使用所有偏移量的平均值作为参考
            avg_offset = avg_all_offset
            
            # 生成偏移特征序列
            prev_feature_val = prev_window[dim]
            offset_feature_val = prev_feature_val * (1 + avg_offset / 100)  # 偏移对特征的影响
            
            # 计算数值变化率
            current_feature_val = current_window[dim]
            if prev_feature_val == 0:
                val_change_rate = 0 if current_feature_val == 0 else 1.0
            else:
                val_change_rate = abs(current_feature_val - offset_feature_val) / prev_feature_val
            val_change_rate = min(val_change_rate, 2.0)  # 限制最大值
            
            # 计算行为一致性指标变化幅度
            prev_consistency = consistency_scores.get(prev_window_id, 0)
            curr_consistency = consistency_scores.get(window_id, 0)
            consistency_change = abs(curr_consistency - prev_consistency)
            
            # 计算临时异常系数
            temp_anomaly_coeff = 0.6 * val_change_rate + 0.3 * energy_diffs[dim] + 0.1 * consistency_change
            
            # 计算异常概率
            if temp_anomaly_coeff == 0:
                normalized_ratio = 0
            else:
                normalized_ratio = min(curr_consistency / temp_anomaly_coeff, 1.0)
            
            anomaly_prob = base_value - normalized_ratio
            anomaly_prob = max(0, min(1, anomaly_prob))  # 确保在0-1之间
            
            anomaly_probabilities[dim][window_id] = anomaly_prob
        
        progress = (i + 1) / total_windows
        update_progress(progress, "计算异常概率")
        if i % 10 == 0:
            time.sleep(0.1)
    
    update_progress(1, "计算异常概率")
    return anomaly_probabilities

def filter_anomaly_features(anomaly_probabilities, threshold=0.6):
    """根据所有特征维度的异常概率筛选异常特征维度"""
    progress = 0
    update_progress(progress, "筛选异常特征")
    
    anomaly_features = {}
    window_ids = sorted(next(iter(anomaly_probabilities.values())).keys())
    total_windows = len(window_ids)
    
    for i, window_id in enumerate(window_ids):
        anomaly_dims = []
        for dim, probs in anomaly_probabilities.items():
            if probs.get(window_id, 0) > threshold:
                anomaly_dims.append(dim)
        
        anomaly_features[window_id] = anomaly_dims
        
        progress = (i + 1) / total_windows
        update_progress(progress, "筛选异常特征")
        if i % 50 == 0:
            time.sleep(0.05)
    
    update_progress(1, "筛选异常特征")
    return anomaly_features