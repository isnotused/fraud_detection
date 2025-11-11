import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import os

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class FraudVisualizer:
    """欺诈检测可视化工具类"""
    
    def __init__(self, output_dir="results"):
        """初始化可视化工具"""
        self.output_dir = output_dir
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def plot_risk_trend(self, risk_values, title="时间窗口动态欺诈风险值趋势"):
        """绘制动态欺诈风险值趋势图"""
        window_ids = sorted(risk_values.keys())
        risks = [risk_values[window_id] for window_id in window_ids]
        
        plt.figure(figsize=(12, 6))
        plt.plot(window_ids, risks, 'b-', marker='o', linewidth=2, markersize=5)
        plt.axhline(y=0.7, color='r', linestyle='--', label='风险阈值')
        plt.title(title, fontsize=15)
        plt.xlabel('时间窗口ID', fontsize=12)
        plt.ylabel('动态欺诈风险值', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 标记高风险窗口
        high_risk_windows = [window_ids[i] for i, r in enumerate(risks) if r > 0.7]
        high_risk_values = [r for r in risks if r > 0.7]
        plt.scatter(high_risk_windows, high_risk_values, color='red', s=50, zorder=5, label='高风险窗口')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "risk_trend.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存风险趋势图到 {output_path}")
        return output_path
    
    def plot_anomaly_features_dist(self, anomaly_features, title="各时间窗口异常特征分布"):
        """绘制异常特征分布热图"""
        window_ids = sorted(anomaly_features.keys())
        all_features = set()
        for features in anomaly_features.values():
            all_features.update(features)
        all_features = sorted(list(all_features))
        
        # 创建特征存在矩阵
        feature_matrix = []
        for window_id in window_ids:
            window_features = anomaly_features[window_id]
            row = [1 if f in window_features else 0 for f in all_features]
            feature_matrix.append(row)
        
        # 转换为DataFrame
        df = pd.DataFrame(feature_matrix, index=window_ids, columns=all_features)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, cmap="YlOrRd", cbar_kws={'label': '是否异常'})
        plt.title(title, fontsize=15)
        plt.xlabel('特征维度', fontsize=12)
        plt.ylabel('时间窗口ID', fontsize=12)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "anomaly_features_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存异常特征热图到 {output_path}")
        return output_path
    
    def plot_behavior_consistency(self, consistency_scores, title="各时间窗口行为一致性指标"):
        """绘制行为一致性指标图"""
        window_ids = sorted(consistency_scores.keys())
        scores = [consistency_scores[window_id] for window_id in window_ids]
        
        plt.figure(figsize=(12, 6))
        plt.bar(window_ids, scores, color='skyblue')
        plt.axhline(y=0.5, color='r', linestyle='--', label='一致性阈值')
        plt.title(title, fontsize=15)
        plt.xlabel('时间窗口ID', fontsize=12)
        plt.ylabel('行为一致性指标', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "behavior_consistency.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存行为一致性指标图到 {output_path}")
        return output_path
    
    def plot_feature_anomaly_prob(self, anomaly_probabilities, title="各特征维度异常概率分布"):
        """绘制各特征维度异常概率分布"""
        window_ids = sorted(next(iter(anomaly_probabilities.values())).keys())
        features = list(anomaly_probabilities.keys())
        
        plt.figure(figsize=(14, 8))
        for feature in features:
            probs = [anomaly_probabilities[feature][window_id] for window_id in window_ids]
            plt.plot(window_ids, probs, marker='.', label=feature)
        
        plt.axhline(y=0.6, color='r', linestyle='--', label='异常阈值')
        plt.title(title, fontsize=15)
        plt.xlabel('时间窗口ID', fontsize=12)
        plt.ylabel('异常概率', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "feature_anomaly_prob.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存特征异常概率分布图到 {output_path}")
        return output_path
    
    def plot_transaction_distribution(self, transaction_data, title="交易类型分布与欺诈关联"):
        """绘制交易类型分布与欺诈关联图"""
        # 按交易类型统计欺诈比例
        trans_type_fraud = transaction_data.groupby('transaction_type')['is_fraud'].agg(['mean', 'count'])
        trans_type_fraud = trans_type_fraud.sort_values('mean', ascending=False)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 柱状图：交易数量
        ax1.bar(trans_type_fraud.index, trans_type_fraud['count'], color='lightgray', label='交易数量')
        ax1.set_xlabel('交易类型', fontsize=12)
        ax1.set_ylabel('交易数量', fontsize=12, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # 折线图：欺诈比例
        ax2 = ax1.twinx()
        ax2.plot(trans_type_fraud.index, trans_type_fraud['mean'], 'ro-', label='欺诈比例')
        ax2.set_ylabel('欺诈比例', fontsize=12, color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, max(trans_type_fraud['mean']) * 1.2)
        
        plt.title(title, fontsize=15)
        fig.tight_layout()
        
        output_path = os.path.join(self.output_dir, "transaction_fraud_correlation.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存交易类型欺诈关联图到 {output_path}")
        return output_path
    
    def plot_user_behavior_graph(self, connection_strength, user_index, title="用户行为交互网络图"):
        """绘制用户行为交互网络图"""
        users = list(user_index.keys())
        num_users = len(users)
        
        # 获取连接强度最高的前50对用户
        edges = []
        for i in range(num_users):
            for j in range(i + 1, num_users):
                if connection_strength[i, j] > 0:
                    edges.append((i, j, connection_strength[i, j]))
        
        # 按连接强度排序并取前50
        edges.sort(key=lambda x: x[2], reverse=True)
        top_edges = edges[:50]
        
        # 绘制网络图
        plt.figure(figsize=(12, 10))
        pos = np.random.rand(num_users, 2)  
        
        # 绘制节点
        plt.scatter(pos[:, 0], pos[:, 1], s=100, c='lightblue', edgecolors='gray')
        
        # 绘制边
        for i, j, strength in top_edges:
            plt.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], 
                     'gray', linewidth=min(strength / 5, 3), alpha=0.7)
        
        plt.title(title, fontsize=15)
        plt.axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "user_behavior_graph.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存用户行为交互网络图到 {output_path}")
        return output_path
    
    def plot_model_performance(self, accuracy_scores, title="模型检测准确率变化"):
        """绘制模型性能变化图"""
        window_ids = range(len(accuracy_scores))
        
        plt.figure(figsize=(12, 6))
        plt.plot(window_ids, accuracy_scores, 'g-', marker='s', linewidth=2, markersize=5)
        plt.title(title, fontsize=15)
        plt.xlabel('模型更新次数', fontsize=12)
        plt.ylabel('检测准确率', fontsize=12)
        plt.ylim(0.5, 1.0)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, "model_performance.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存模型性能变化图到 {output_path}")
        return output_path