import numpy as np
import pandas as pd
from scipy.signal import correlate
from progress_bar import update_progress
import time

def calculate_time_offset(transaction_sequences):
    """确定不同交易数据序列之间的时序偏移量"""
    progress = 0
    update_progress(progress, "计算时序偏移量")
    
    offsets = {}
    sequences = list(transaction_sequences.items())
    total_pairs = len(sequences) * (len(sequences) - 1) // 2
    pair_count = 0
    
    # 预设初始值和迭代增量
    initial_offset = 0
    iteration_increment = 1
    
    for i in range(len(sequences)):
        seq1_id, seq1_data = sequences[i]
        for j in range(i + 1, len(sequences)):
            seq2_id, seq2_data = sequences[j]
            
            # 确保序列长度一致
            min_len = min(len(seq1_data), len(seq2_data))
            seq1 = seq1_data[:min_len]
            seq2 = seq2_data[:min_len]
            
            # 利用数值差异构建互相关函数
            diffs1 = np.diff(seq1)
            diffs2 = np.diff(seq2)
            
            # 计算互相关
            corr = correlate(diffs1, diffs2, mode='same')
            
            # 找到最大相关值对应的偏移量
            max_corr_idx = np.argmax(np.abs(corr))
            optimal_offset = max_corr_idx - len(corr) // 2
            
            # 迭代优化偏移量
            current_offset = initial_offset
            best_corr = -np.inf
            best_offset = current_offset
            
            # 搜索附近的偏移量
            for offset in range(current_offset - 5, current_offset + 6, iteration_increment):
                if abs(offset) >= len(seq1) // 2:
                    continue
                
                if offset >= 0:
                    shifted_seq1 = seq1[offset:]
                    shifted_seq2 = seq2[:len(shifted_seq1)]
                else:
                    shifted_seq1 = seq1[:len(seq2) + offset]
                    shifted_seq2 = seq2[-offset:]
                
                if len(shifted_seq1) == 0 or len(shifted_seq2) == 0:
                    continue
                
                current_corr = np.corrcoef(shifted_seq1, shifted_seq2)[0, 1]
                
                if current_corr > best_corr:
                    best_corr = current_corr
                    best_offset = offset
            
            # 综合两种方法的结果
            final_offset = best_offset if abs(best_corr) > 0.1 else optimal_offset
            
            offsets[(seq1_id, seq2_id)] = final_offset
            
            pair_count += 1
            progress = pair_count / total_pairs
            update_progress(progress, "计算时序偏移量")
            if pair_count % 5 == 0:
                time.sleep(0.1)
    
    update_progress(1, "计算时序偏移量")
    return offsets

def build_user_behavior_graph(user_behavior_data, transaction_data):
    """构建用户行为交互图"""
    progress = 0
    update_progress(progress, "构建用户行为交互图")
    
    # 获取所有用户
    users = user_behavior_data["user_id"].unique()
    user_index = {user: i for i, user in enumerate(users)}
    num_users = len(users)
    
    # 初始化连接强度矩阵
    connection_strength = np.zeros((num_users, num_users))
    
    # 按时间窗口分析用户间的交易连接
    windows = transaction_data["window_id"].unique()
    total_windows = len(windows)
    
    for i, window_id in enumerate(windows):
        window_trans = transaction_data[transaction_data["window_id"] == window_id]
        
        # 对于转账类型，建立用户间的连接
        transfers = window_trans[window_trans["transaction_type"] == "转账"]
        
        for _, trans in transfers.iterrows():
            other_users = [u for u in users if u != trans["user_id"]]
            if other_users:
                recipient = np.random.choice(other_users)
                sender_idx = user_index[trans["user_id"]]
                recipient_idx = user_index[recipient]
                
                # 连接强度与交易金额正相关
                connection_strength[sender_idx, recipient_idx] += trans["amount"] / 1000  # 归一化
                connection_strength[recipient_idx, sender_idx] += trans["amount"] / 1000  # 双向连接
        
        progress = (i + 1) / total_windows
        update_progress(progress, "构建用户行为交互图")
        if i % 10 == 0:
            time.sleep(0.05)
    
    update_progress(1, "构建用户行为交互图")
    return connection_strength, user_index

def generate_behavior_feature_distribution(connection_strength, user_index, window_features, user_behavior_data):
    """生成用户行为特征分布曲线"""
    progress = 0
    update_progress(progress, "生成行为特征分布")
    
    # 为每个窗口生成行为特征分布
    windows = sorted(window_features["window_id"].unique())
    total_windows = len(windows)
    behavior_distributions = {}
    
    for i, window_id in enumerate(windows):
        # 计算每个用户的行为活跃度
        user_activity = {}
        for user, idx in user_index.items():
            # 连接强度总和表示用户的活跃度
            activity = np.sum(connection_strength[idx, :])
            
            # 结合用户自身特征
            user_info = user_behavior_data[user_behavior_data["user_id"] == user].iloc[0]
            credit_factor = user_info["credit_score"] / 850  # 信用分数归一化
            age_factor = 1 - abs(user_info["age"] - 35) / 50  
            
            # 综合行为特征
            user_activity[user] = activity * credit_factor * age_factor
        
        # 转换为分布
        activity_values = np.array(list(user_activity.values()))
        if len(activity_values) > 0:
            activity_dist = activity_values / np.sum(activity_values)
            behavior_distributions[window_id] = activity_dist
        else:
            behavior_distributions[window_id] = np.array([])
        
        progress = (i + 1) / total_windows
        update_progress(progress, "生成行为特征分布")
        if i % 10 == 0:
            time.sleep(0.05)
    
    update_progress(1, "生成行为特征分布")
    return behavior_distributions

def calculate_behavior_consistency(transaction_distributions, behavior_distributions, time_offsets):
    """计算每个时间窗口的行为一致性指标"""
    progress = 0
    update_progress(progress, "计算行为一致性指标")
    
    consistency_scores = {}
    window_ids = sorted(transaction_distributions.keys())
    total_windows = len(window_ids)
    
    # 计算所有偏移量的平均影响
    all_offsets = [abs(offset) for offset in time_offsets.values()]
    avg_offset = np.mean(all_offsets) if all_offsets else 0
    base_offset_impact = min(avg_offset / 100, 1.0)  # 基础偏移影响
    
    for i, window_id in enumerate(window_ids):
        # 获取当前窗口的交易特征分布和行为特征分布
        trans_dist = transaction_distributions.get(window_id, np.array([]))
        behav_dist = behavior_distributions.get(window_id, np.array([]))
        
        # 确保分布长度一致
        min_len = min(len(trans_dist), len(behav_dist))
        if min_len == 0:
            consistency_scores[window_id] = 0.0
            continue
        
        trans_dist = trans_dist[:min_len]
        behav_dist = behav_dist[:min_len]
        
        # 计算分布相似度 - 使用余弦相似度
        dot_product = np.dot(trans_dist, behav_dist)
        norm_trans = np.linalg.norm(trans_dist)
        norm_behav = np.linalg.norm(behav_dist)
        
        if norm_trans == 0 or norm_behav == 0:
            similarity = 0.0
        else:
            similarity = dot_product / (norm_trans * norm_behav)
        
        # 使用基础偏移影响，避免类型比较错误
        offset_impact = base_offset_impact
        
        # 综合考虑相似度和偏移影响
        consistency = similarity * (1 - offset_impact)
        consistency_scores[window_id] = max(0, min(1, consistency))  # 确保在0-1之间
        
        progress = (i + 1) / total_windows
        update_progress(progress, "计算行为一致性指标")
        if i % 10 == 0:
            time.sleep(0.05)
    
    update_progress(1, "计算行为一致性指标")
    return consistency_scores