import pandas as pd
import numpy as np
from progress_bar import update_progress
import time

def segment_data_by_time_window(transaction_data, window_size="1H"):
    """根据预设的时间窗口对金融消费交易数据进行分段处理"""
    progress = 0
    update_progress(progress, "数据分段处理")
    
    # 确保数据按时间排序
    transaction_data = transaction_data.sort_values("transaction_time")
    
    # 创建时间窗口
    start_time = transaction_data["transaction_time"].min()
    end_time = transaction_data["transaction_time"].max()
    
    # 生成所有时间窗口
    windows = []
    current_start = start_time
    while current_start < end_time:
        current_end = current_start + pd.Timedelta(window_size)
        windows.append((current_start, current_end))
        current_start = current_end
    
    # 为每个交易分配时间窗口ID
    transaction_data["window_id"] = -1
    total_windows = len(windows)
    
    for i, (start, end) in enumerate(windows):
        mask = (transaction_data["transaction_time"] >= start) & (transaction_data["transaction_time"] < end)
        transaction_data.loc[mask, "window_id"] = i
        
        progress = (i + 1) / total_windows
        update_progress(progress, "数据分段处理")
        # 添加延迟以控制进度
        if i % 10 == 0:
            time.sleep(0.05)
    
    update_progress(1, "数据分段处理")
    return transaction_data, windows

def extract_transaction_features(segmented_data):
    """提取交易特征"""
    progress = 0
    update_progress(progress, "提取交易特征")
    
    # 按窗口和用户分组计算特征
    window_features = []
    windows = segmented_data["window_id"].unique()
    total_windows = len(windows)
    
    for i, window_id in enumerate(windows):
        window_data = segmented_data[segmented_data["window_id"] == window_id]
        
        # 窗口级特征
        total_transactions = len(window_data)
        total_amount = window_data["amount"].sum()
        avg_amount = window_data["amount"].mean() if total_transactions > 0 else 0
        max_amount = window_data["amount"].max() if total_transactions > 0 else 0
        min_amount = window_data["amount"].min() if total_transactions > 0 else 0
        
        # 交易类型分布
        trans_type_counts = window_data["transaction_type"].value_counts(normalize=True).to_dict()
        
        # 位置分布
        location_counts = window_data["location"].value_counts(normalize=True).to_dict()
        
        # 欺诈标记计数
        fraud_count = window_data["is_fraud"].sum()
        fraud_ratio = fraud_count / total_transactions if total_transactions > 0 else 0
        
        window_features.append({
            "window_id": window_id,
            "start_time": window_data["transaction_time"].min() if total_transactions > 0 else None,
            "end_time": window_data["transaction_time"].max() if total_transactions > 0 else None,
            "total_transactions": total_transactions,
            "total_amount": total_amount,
            "avg_amount": avg_amount,
            "max_amount": max_amount,
            "min_amount": min_amount,
            "trans_type_dist": trans_type_counts,
            "location_dist": location_counts,
            "fraud_count": fraud_count,
            "fraud_ratio": fraud_ratio
        })
        
        progress = (i + 1) / total_windows
        update_progress(progress, "提取交易特征")
        if i % 10 == 0:
            time.sleep(0.05)
    
    features_df = pd.DataFrame(window_features)
    update_progress(1, "提取交易特征")
    return features_df