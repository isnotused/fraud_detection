import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import normalize

class BehaviorConsistencyAnalyzer:
    """行为一致性分析器，用于计算时间窗口的行为一致性指标"""
    
    def __init__(self):
        """初始化行为一致性分析器"""
        pass
        
    def calculate_distribution_similarity(self, dist1, dist2):
        """计算两个分布之间的相似性
        
        参数:
            dist1: 第一个分布
            dist2: 第二个分布
            
        返回:
            分布相似性分数，范围[0, 1]，值越大表示越相似
        """
        # 确保两个分布具有相同的长度
        min_length = min(len(dist1), len(dist2))
        dist1 = dist1[:min_length]
        dist2 = dist2[:min_length]
        
        # 标准化分布
        dist1_norm = normalize([dist1])[0]
        dist2_norm = normalize([dist2])[0]
        
        # 计算多种相似性度量并取平均值
        cosine_sim = np.dot(dist1_norm, dist2_norm) / (np.linalg.norm(dist1_norm) * np.linalg.norm(dist2_norm))
        
        # 计算KS检验统计量（越小表示分布越相似）
        ks_statistic, _ = ks_2samp(dist1, dist2)
        ks_sim = 1 - ks_statistic  # 转换为相似性分数
        
        # 计算互信息（需要离散化）
        bins = min(10, len(dist1))  # 最大10个 bins
        hist1, _ = np.histogram(dist1, bins=bins, density=True)
        hist2, _ = np.histogram(dist2, bins=bins, density=True)
        mi_score = mutual_info_score(hist1, hist2)
        mi_sim = mi_score / np.log(bins)  # 归一化到[0, 1]
        
        # 综合三种相似性度量，取平均值
        similarity = (cosine_sim + ks_sim + mi_sim) / 3
        
        # 确保结果在[0, 1]范围内
        return max(0, min(1, similarity))
        
    def calculate_window_consistency(self, transaction_dist, behavior_dist):
        """计算单个时间窗口的行为一致性指标
        
        参数:
            transaction_dist: 交易特征分布
            behavior_dist: 用户行为特征分布（已进行时序偏移处理）
            
        返回:
            行为一致性指标，范围[0, 1]，值越大表示一致性越高
        """
        if len(transaction_dist) == 0 or len(behavior_dist) == 0:
            return 0.0  # 空分布返回最低一致性
            
        return self.calculate_distribution_similarity(transaction_dist, behavior_dist)
        
    def calculate_consistency_sequence(self, transaction_distributions, behavior_distributions, offsets):
        """计算多个时间窗口的行为一致性指标序列
        
        参数:
            transaction_distributions: 各窗口的交易特征分布列表
            behavior_distributions: 各窗口的用户行为特征分布列表
            offsets: 各窗口对应的时序偏移量列表
            
        返回:
            每个窗口的行为一致性指标列表
        """
        num_windows = len(transaction_distributions)
        consistency_scores = []
        
        for i in range(num_windows):
            # 获取当前窗口的交易分布
            trans_dist = transaction_distributions[i]
            
            # 根据偏移量获取对应的行为分布
            behav_idx = i + offsets[i] if 0 <= i + offsets[i] < len(behavior_distributions) else i
            behav_dist = behavior_distributions[behav_idx]
            
            # 计算一致性分数
            score = self.calculate_window_consistency(trans_dist, behav_dist)
            consistency_scores.append(score)
            
        return consistency_scores
        
    def get_consistency_changes(self, consistency_scores):
        """计算行为一致性指标的变化量
        
        参数:
            consistency_scores: 行为一致性指标序列
            
        返回:
            每个窗口相对于前一个窗口的一致性变化量列表
        """
        if len(consistency_scores) <= 1:
            return [0.0] * len(consistency_scores)
            
        changes = [0.0]  # 第一个窗口没有前一个窗口，变化量为0
        for i in range(1, len(consistency_scores)):
            change = abs(consistency_scores[i] - consistency_scores[i-1])
            changes.append(change)
            
        return changes
