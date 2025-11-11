import pandas as pd
import numpy as np
from datetime import timedelta

class TimeWindowProcessor:
    """时间窗口处理器，负责将数据按预设时间窗口进行分段处理"""
    
    def __init__(self, window_size=60, slide_step=30):
        """初始化时间窗口处理器
        
        参数:
            window_size: 时间窗口大小，单位为分钟
            slide_step: 滑动步长，单位为分钟
        """
        self.window_size = window_size
        self.slide_step = slide_step
        
    def get_time_windows(self, start_time, end_time):
        """生成从开始时间到结束时间的所有时间窗口
        
        参数:
            start_time: 开始时间
            end_time: 结束时间
            
        返回:
            时间窗口列表，每个元素是一个包含开始和结束时间的元组
        """
        windows = []
        current_start = start_time
        
        # 计算窗口时间增量
        window_delta = timedelta(minutes=self.window_size)
        slide_delta = timedelta(minutes=self.slide_step)
        
        while current_start + window_delta <= end_time:
            current_end = current_start + window_delta
            windows.append((current_start, current_end))
            current_start += slide_delta
            
        return windows
    
    def segment_transaction_data(self, transaction_data):
        """将交易数据按时间窗口分段
        
        参数:
            transaction_data: 交易数据DataFrame，需包含'transaction_time'列
            
        返回:
            按时间窗口分段的交易数据字典，键为窗口索引，值为该窗口内的交易数据
        """
        if transaction_data.empty:
            return {}
            
        # 确定整体时间范围
        start_time = transaction_data['transaction_time'].min()
        end_time = transaction_data['transaction_time'].max()
        
        # 获取所有时间窗口
        windows = self.get_time_windows(start_time, end_time)
        
        # 按窗口分割数据
        segmented_data = {}
        for i, (window_start, window_end) in enumerate(windows):
            mask = (transaction_data['transaction_time'] >= window_start) & \
                   (transaction_data['transaction_time'] < window_end)
            segmented_data[i] = transaction_data[mask]
            
        return segmented_data
    
    def segment_behavior_data(self, behavior_data):
        """将用户行为数据按时间窗口分段
        
        参数:
            behavior_data: 用户行为数据DataFrame，需包含'behavior_time'列
            
        返回:
            按时间窗口分段的用户行为数据字典，键为窗口索引，值为该窗口内的行为数据
        """
        if behavior_data.empty:
            return {}
            
        # 确定整体时间范围
        start_time = behavior_data['behavior_time'].min()
        end_time = behavior_data['behavior_time'].max()
        
        # 获取所有时间窗口
        windows = self.get_time_windows(start_time, end_time)
        
        # 按窗口分割数据
        segmented_data = {}
        for i, (window_start, window_end) in enumerate(windows):
            mask = (behavior_data['behavior_time'] >= window_start) & \
                   (behavior_data['behavior_time'] < window_end)
            segmented_data[i] = behavior_data[mask]
            
        return segmented_data
        
    def segment_user_data(self, combined_user_data):
        """将用户的交易和行为数据按时间窗口同步分段
        
        参数:
            combined_user_data: 包含用户交易和行为数据的字典
            
        返回:
            按时间窗口组织的用户数据字典
        """
        transactions = combined_user_data['transactions']
        behavior = combined_user_data['behavior']
        
        if transactions.empty and behavior.empty:
            return {}
            
        # 确定整体时间范围（取两者的并集）
        if transactions.empty:
            start_time = behavior['behavior_time'].min()
            end_time = behavior['behavior_time'].max()
        elif behavior.empty:
            start_time = transactions['transaction_time'].min()
            end_time = transactions['transaction_time'].max()
        else:
            start_time = min(transactions['transaction_time'].min(), 
                           behavior['behavior_time'].min())
            end_time = max(transactions['transaction_time'].max(), 
                         behavior['behavior_time'].max())
        
        # 获取所有时间窗口
        windows = self.get_time_windows(start_time, end_time)
        
        # 按窗口分割数据
        user_window_data = {}
        for i, (window_start, window_end) in enumerate(windows):
            # 提取窗口内的交易数据
            trans_mask = (transactions['transaction_time'] >= window_start) & \
                         (transactions['transaction_time'] < window_end)
            window_trans = transactions[trans_mask]
            
            # 提取窗口内的行为数据
            behav_mask = (behavior['behavior_time'] >= window_start) & \
                         (behavior['behavior_time'] < window_end)
            window_behav = behavior[behav_mask]
            
            user_window_data[i] = {
                'transactions': window_trans,
                'behavior': window_behav,
                'window_start': window_start,
                'window_end': window_end
            }
            
        return user_window_data
        
    def get_window_overlap(self, window1, window2):
        """计算两个窗口的重叠时间
        
        参数:
            window1: 第一个窗口，包含开始和结束时间的元组
            window2: 第二个窗口，包含开始和结束时间的元组
            
        返回:
            重叠的时间长度（分钟），如果没有重叠则返回0
        """
        start1, end1 = window1
        start2, end2 = window2
        
        # 计算重叠的开始和结束时间
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        # 如果没有重叠，返回0
        if overlap_start >= overlap_end:
            return 0
            
        # 计算重叠分钟数
        overlap_delta = overlap_end - overlap_start
        return overlap_delta.total_seconds() / 60
        
    def get_window_by_index(self, user_window_data, window_index):
        """根据索引获取特定窗口的数据
        
        参数:
            user_window_data: 按窗口组织的用户数据
            window_index: 窗口索引
            
        返回:
            指定窗口的数据，如果不存在则返回None
        """
        return user_window_data.get(window_index, None)
