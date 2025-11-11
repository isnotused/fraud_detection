import numpy as np
import pandas as pd

class TemporalOffsetAnalyzer:
    """时序偏移分析器，用于计算不同交易数据序列之间的时序偏移量"""
    
    def __init__(self, max_offset=100, step=1):
        """初始化时序偏移分析器
        
        参数:
            max_offset: 最大偏移量
            step: 迭代增量
        """
        self.max_offset = max_offset
        self.step = step
        
    def calculate_cross_correlation(self, sequence1, sequence2):
        """计算两个序列之间的互相关函数
        
        参数:
            sequence1: 第一个序列
            sequence2: 第二个序列
            
        返回:
            互相关函数值数组
        """
        # 确保两个序列长度相同
        min_length = min(len(sequence1), len(sequence2))
        sequence1 = sequence1[:min_length]
        sequence2 = sequence2[:min_length]
        
        # 计算互相关
        correlation = np.correlate(sequence1, sequence2, mode='same')
        return correlation
        
    def find_optimal_offset(self, sequence1, sequence2):
        """找到两个序列之间的最优时序偏移量
        
        参数:
            sequence1: 第一个序列
            sequence2: 第二个序列
            
        返回:
            最优时序偏移量
        """
        # 如果序列为空或长度不足，返回0偏移
        if len(sequence1) < 2 or len(sequence2) < 2:
            return 0
            
        # 标准化序列
        seq1_norm = (sequence1 - np.mean(sequence1)) / np.std(sequence1)
        seq2_norm = (sequence2 - np.mean(sequence2)) / np.std(sequence2)
        
        max_correlation = -np.inf
        optimal_offset = 0
        
        # 迭代测试不同的偏移量
        for offset in range(-self.max_offset, self.max_offset + 1, self.step):
            # 根据偏移量调整序列
            if offset > 0:
                adjusted_seq1 = seq1_norm[offset:]
                adjusted_seq2 = seq2_norm[:-offset]
            elif offset < 0:
                adjusted_seq1 = seq1_norm[:offset]
                adjusted_seq2 = seq2_norm[-offset:]
            else:
                adjusted_seq1 = seq1_norm
                adjusted_seq2 = seq2_norm
                
            # 计算调整后的互相关
            correlation = np.mean(adjusted_seq1 * adjusted_seq2)
            
            # 更新最大相关值和对应的偏移量
            if correlation > max_correlation:
                max_correlation = correlation
                optimal_offset = offset
                
        return optimal_offset
        
    def find_offsets_between_features(self, feature_sequences):
        """计算多个特征序列之间的时序偏移量
        
        参数:
            feature_sequences: 特征序列字典，键为特征名称，值为序列数组
            
        返回:
            偏移量矩阵，offset_matrix[i][j]表示第i个序列相对于第j个序列的偏移量
        """
        features = list(feature_sequences.keys())
        num_features = len(features)
        
        # 初始化偏移量矩阵
        offset_matrix = np.zeros((num_features, num_features), dtype=int)
        
        # 计算每对序列之间的偏移量
        for i in range(num_features):
            for j in range(num_features):
                if i != j:
                    seq_i = feature_sequences[features[i]]
                    seq_j = feature_sequences[features[j]]
                    offset = self.find_optimal_offset(seq_i, seq_j)
                    offset_matrix[i][j] = offset
                    
        return offset_matrix, features
        
    def apply_offset(self, sequence, offset):
        """将偏移量应用到序列上
        
        参数:
            sequence: 原始序列
            offset: 要应用的偏移量
            
        返回:
            偏移后的序列
        """
        if offset == 0:
            return sequence
            
        # 正偏移表示序列向右移动（滞后）
        # 负偏移表示序列向左移动（超前）
        if offset > 0:
            # 向右移动，前面补0
            return np.concatenate([np.zeros(offset), sequence[:-offset]])
        else:
            # 向左移动，后面补0
            return np.concatenate([sequence[-offset:], np.zeros(-offset)])
            
    def align_sequences(self, feature_sequences, reference_feature):
        """将所有特征序列与参考特征序列对齐
        
        参数:
            feature_sequences: 特征序列字典
            reference_feature: 参考特征名称
            
        返回:
            对齐后的特征序列字典
        """
        if reference_feature not in feature_sequences:
            raise ValueError(f"参考特征 {reference_feature} 不在特征序列中")
            
        aligned_sequences = {}
        reference_seq = feature_sequences[reference_feature]
        
        for feature, sequence in feature_sequences.items():
            if feature == reference_feature:
                aligned_sequences[feature] = sequence
            else:
                offset = self.find_optimal_offset(sequence, reference_seq)
                aligned_sequences[feature] = self.apply_offset(sequence, offset)
                
        return aligned_sequences
