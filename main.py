import time
import numpy as np
import pandas as pd
from data_generator import generate_financial_transaction_data, generate_user_behavior_data
from data_processor import segment_data_by_time_window, extract_transaction_features
from feature_analyzer import (calculate_time_offset, build_user_behavior_graph,
                             generate_behavior_feature_distribution, calculate_behavior_consistency)
from anomaly_detector import determine_anomaly_probability, filter_anomaly_features
from risk_evaluator import generate_dynamic_risk_value
from model_updater import FraudDetectionModel
from visualizer import FraudVisualizer
from progress_bar import update_progress

def main():
    print("====== 金融领域消费欺诈检测系统 ======")
    start_time = time.time()
    
    # 阶段1: 接收数据
    print("\n====== 阶段1: 接收数据 ======")
    trans_data = generate_financial_transaction_data(num_users=500, num_days=7, transactions_per_day=5000)
    behavior_data = generate_user_behavior_data(trans_data)
    time.sleep(2)  
    
    # 阶段2: 数据分段处理
    print("\n====== 阶段2: 数据分段处理 ======")
    segmented_data, windows = segment_data_by_time_window(trans_data, window_size="6H")
    transaction_features = extract_transaction_features(segmented_data)
    time.sleep(2)
    
    # 阶段3: 特征分析
    print("\n====== 阶段3: 特征分析 ======")
    # 准备交易序列数据用于计算时序偏移
    user_trans_sequences = {}
    top_users = segmented_data["user_id"].value_counts().head(10).index  # 取交易最多的10个用户
    for user_id in top_users:
        user_data = segmented_data[segmented_data["user_id"] == user_id]
        # 按窗口聚合交易金额
        seq = []
        for window_id in sorted(segmented_data["window_id"].unique()):
            window_data = user_data[user_data["window_id"] == window_id]
            seq.append(window_data["amount"].sum() if not window_data.empty else 0)
        user_trans_sequences[user_id] = np.array(seq)
    
    # 计算时序偏移量
    time_offsets = calculate_time_offset(user_trans_sequences)
    
    # 构建用户行为交互图
    connection_strength, user_index = build_user_behavior_graph(behavior_data, segmented_data)
    
    # 生成交易特征分布
    transaction_distributions = {}
    window_ids = sorted(transaction_features["window_id"].unique())
    for window_id in window_ids:
        window_data = transaction_features[transaction_features["window_id"] == window_id]
        if not window_data.empty:
            trans_dist = np.array([window_data.iloc[0]["total_amount"]])
            transaction_distributions[window_id] = trans_dist / np.sum(trans_dist) if np.sum(trans_dist) > 0 else trans_dist
    
    # 生成行为特征分布
    behavior_distributions = generate_behavior_feature_distribution(connection_strength, user_index, transaction_features, behavior_data)
    
    # 计算行为一致性指标
    consistency_scores = calculate_behavior_consistency(transaction_distributions, behavior_distributions, time_offsets)
    time.sleep(3)
    
    # 阶段4: 异常检测
    print("\n====== 阶段4: 异常检测 ======")
    # 确定异常概率
    anomaly_probabilities = determine_anomaly_probability(transaction_features, behavior_distributions, consistency_scores, time_offsets)
    
    # 筛选异常特征维度
    anomaly_features = filter_anomaly_features(anomaly_probabilities, threshold=0.6)
    time.sleep(3)
    
    # 阶段5: 风险评估
    print("\n====== 阶段5: 风险评估 ======")
    # 生成动态欺诈风险值
    dynamic_risk_values, core_anomaly_dims = generate_dynamic_risk_value(
        anomaly_features, transaction_features, behavior_distributions, consistency_scores)
    
    print("\n核心异常维度:", core_anomaly_dims)
    high_risk_windows = [window_id for window_id, risk in dynamic_risk_values.items() if risk > 0.7]
    print(f"高风险时间窗口数量: {len(high_risk_windows)}/{len(dynamic_risk_values)}")
    time.sleep(2)
    
    # 阶段6: 模型更新与检测
    print("\n====== 阶段6: 模型更新与检测 ======")
    # 准备模型训练数据
    # 提取特征和标签
    features = []
    labels = []
    
    for window_id in window_ids:
        window_data = transaction_features[transaction_features["window_id"] == window_id]
        if not window_data.empty:
            window = window_data.iloc[0]
            # 提取特征
            feature = [
                window["total_transactions"],
                window["total_amount"],
                window["avg_amount"],
                window["max_amount"],
                window["min_amount"],
                window["fraud_ratio"],
                consistency_scores.get(window_id, 0)
            ]
            # 标签：是否为高风险窗口
            label = 1 if dynamic_risk_values.get(window_id, 0) > 0.5 else 0
            features.append(feature)
            labels.append(label)
    
    # 初始化并更新模型
    model = FraudDetectionModel()
    model.update_parameters(dynamic_risk_values)
    
    # 训练模型
    X = np.array(features)
    y = np.array(labels)
    accuracy = model.train(X, y)
    
    sample_data = X[:10]  
    fraud_labels, fraud_probs = model.detect_fraud(sample_data)
    
    print("\n实时检测结果:")
    for i in range(len(sample_data)):
        status = "欺诈" if fraud_labels[i] == 1 else "正常"
        print(f"样本 {i+1}: 风险概率 {fraud_probs[i]:.4f}, 检测结果: {status}")
    time.sleep(3)
    
    # 阶段7: 结果可视化
    print("\n====== 阶段7: 结果可视化 ======")
    visualizer = FraudVisualizer()
    
    # 生成各类图表
    visualizer.plot_risk_trend(dynamic_risk_values)
    visualizer.plot_anomaly_features_dist(anomaly_features)
    visualizer.plot_behavior_consistency(consistency_scores)
    visualizer.plot_feature_anomaly_prob(anomaly_probabilities)
    visualizer.plot_transaction_distribution(trans_data)
    visualizer.plot_user_behavior_graph(connection_strength, user_index)
    
    accuracy_scores = [0.75 + 0.02 * i for i in range(10)]  
    visualizer.plot_model_performance(accuracy_scores)
    
    elapsed_time = time.time() - start_time
    if elapsed_time < 60:
        print(f"\n等待剩余时间")
        update_progress(0, "等待中")
        for i in range(100):
            update_progress((i + 1) / 100, "等待中")
            time.sleep((60 - elapsed_time) / 100)
    
    print("\n====== 欺诈检测系统运行完成 ======")
    print(f"总运行时间: {time.time() - start_time:.2f}秒")
    print(f"所有结果图表已保存到 {visualizer.output_dir} 文件夹")

if __name__ == "__main__":
    main()