import numpy as np
import pandas as pd
from progress_bar import update_progress
import time

def generate_dynamic_risk_value(anomaly_features, transaction_features, behavior_distributions, consistency_scores):
    """根据所有时间窗口的异常特征维度分布生成动态欺诈风险值"""
    progress = 0
    update_progress(progress, "生成欺诈风险值")
    
    window_ids = sorted(transaction_features["window_id"].unique())
    total_windows = len(window_ids)
    risk_values = {}
    
    # 收集所有窗口的异常特征
    all_anomaly_features = [set(features) for features in anomaly_features.values()]
    
    # 计算核心异常维度（所有窗口异常特征的交集）
    if all_anomaly_features:
        core_anomaly_dims = set.intersection(*[f for f in all_anomaly_features if f])
    else:
        core_anomaly_dims = set()
    
    # 为每个窗口计算风险值
    for i, window_id in enumerate(window_ids):
        # 获取当前窗口的异常特征
        window_anomalies = set(anomaly_features.get(window_id, []))
        
        # 计算核心异常维度数值分布方差
        core_dims = list(core_anomaly_dims & window_anomalies)
        if core_dims:
            # 获取核心维度的数值
            core_values = []
            for dim in core_dims:
                window_data = transaction_features[transaction_features["window_id"] == window_id]
                if not window_data.empty:
                    core_values.append(window_data.iloc[0][dim])
            
            if core_values:
                core_variance = np.var(core_values)
            else:
                core_variance = 0
        else:
            core_variance = 0
        
        # 计算用户行为轨迹变化量（与前一窗口比较）
        if window_id > 0:
            behavior_change = calculate_user_behavior_change(behavior_distributions, window_id - 1, window_id)
        else:
            behavior_change = 0
        
        # 计算欺诈风险因子（加权结果）
        consistency_score = consistency_scores.get(window_id, 0)
        # 一致性越低，风险权重越高
        consistency_weight = 1 - consistency_score
        
        # 计算风险因子
        risk_factor = (0.7 * core_variance * consistency_weight) + (0.3 * behavior_change)
        
        # 归一化处理
        max_possible_variance = 1e6  # 预设最大可能方差
        normalized_risk = min(risk_factor / max_possible_variance, 1.0)
        
        # 结合异常特征数量调整
        anomaly_count = len(window_anomalies)
        feature_count = len(transaction_features.columns) - 3  # 排除窗口ID和时间列
        anomaly_ratio = anomaly_count / feature_count if feature_count > 0 else 0
        
        # 最终风险值
        final_risk = min(normalized_risk + anomaly_ratio * 0.5, 1.0)
        risk_values[window_id] = final_risk
        
        progress = (i + 1) / total_windows
        update_progress(progress, "生成欺诈风险值")
        if i % 10 == 0:
            time.sleep(0.05)
    
    update_progress(1, "生成欺诈风险值")
    return risk_values, core_anomaly_dims

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
    
    # 计算JS散度作为变化量
    eps = 1e-10
    avg_dist = (dist1 + dist2) / 2
    kl1 = np.sum(dist1 * np.log((dist1 + eps) / (avg_dist + eps)))
    kl2 = np.sum(dist2 * np.log((dist2 + eps) / (avg_dist + eps)))
    js_div = (kl1 + kl2) / 2
    return min(js_div, 1.0)  # 限制最大值并归一化