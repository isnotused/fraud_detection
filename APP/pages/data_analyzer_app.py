# 风险分析评估
import numpy as np
import pandas as pd
from scipy.signal import correlate
import time
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

data_title = ['欺诈检测结果', '异常特征分布', '风险值趋势', '交易类型分析']


def calculate_time_offset(transaction_sequences):
    """确定不同交易数据序列之间的时序偏移量"""
    
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
            
            # 检查数组是否为空，避免correlate函数出错
            if len(diffs1) == 0 or len(diffs2) == 0:
                offsets[(seq1_id, seq2_id)] = 0
                pair_count += 1
                continue
            
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
            if pair_count % 5 == 0:
                time.sleep(0.1)
    
    return offsets

def build_user_behavior_graph(user_behavior_data, transaction_data):
    """构建用户行为交互图"""
    
    # 获取所有用户
    users = user_behavior_data["user_id"].unique()
    user_index = {user: i for i, user in enumerate(users)}
    num_users = len(users)
    
    # 初始化连接强度矩阵
    connection_strength = np.zeros((num_users, num_users))
    
    # 按时间窗口分析用户间的交易连接
    if "window_id" not in transaction_data.columns:
        # 如果没有window_id，跳过窗口分析
        return connection_strength, user_index
    
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
        
        if i % 10 == 0:
            time.sleep(0.05)
    
    return connection_strength, user_index

def generate_behavior_feature_distribution(connection_strength, user_index, window_features, user_behavior_data):
    """生成用户行为特征分布曲线"""

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
        if i % 10 == 0:
            time.sleep(0.05)
    
    return behavior_distributions

def calculate_behavior_consistency(transaction_distributions, behavior_distributions, time_offsets):
    """计算每个时间窗口的行为一致性指标"""
    
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
        
        if i % 10 == 0:
            time.sleep(0.05)
    
    return consistency_scores

def generate_dynamic_risk_value(anomaly_features, transaction_features, behavior_distributions, consistency_scores):
    """根据所有时间窗口的异常特征维度分布生成动态欺诈风险值"""
    progress = 0
    
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
        
        if i % 10 == 0:
            time.sleep(0.05)
    
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

def tit_button(index):
    """按钮点击回调函数"""
    st.session_state.analyzer_index = index
    st.session_state.analyzer_info = data_title[index]

def data_analyzer_app():
    st.title("🔍 风险分析评估")
    
    # 检查是否有数据可用
    if 'user_data_generated' not in st.session_state or not st.session_state.user_data_generated:
        st.warning("⚠️ 请先在【用户数据】页面查看数据")
        return
    
    # 初始化session state
    if 'analyzer_index' not in st.session_state:
        st.session_state.analyzer_index = 0
        st.session_state.analyzer_info = data_title[0]
    
    # 自动执行风险分析
    if 'analysis_completed' not in st.session_state or not st.session_state.analysis_completed:
        with st.spinner("正在进行风险分析..."):
            perform_risk_analysis()
        st.session_state.analysis_completed = True
    
    tp = lambda x: 'primary' if st.session_state.analyzer_index == x else 'secondary'
    
    col01, col02 = st.columns([1, 5])
    
    with col01:
        st.markdown("### 📋 分析视图")
        with st.container(height=600, border=True):
            for ind, tit in enumerate(data_title):
                if st.button(label=tit, key=f'analyzer_tit_{ind}', use_container_width=True, 
                           on_click=tit_button, args=(ind,), type=tp(ind)):
                    pass
            
            st.divider()
            if st.button("🔄 刷新分析", use_container_width=True):
                st.session_state.analysis_completed = False
                st.rerun()
    
    with col02:
        with st.container(border=True, height=600):
            if st.session_state.analyzer_index == 0:
                show_fraud_detection_results()
            elif st.session_state.analyzer_index == 1:
                show_anomaly_distribution()
            elif st.session_state.analyzer_index == 2:
                show_risk_trend()
            elif st.session_state.analyzer_index == 3:
                show_transaction_analysis()

def perform_risk_analysis():
    """执行风险分析"""
    # 获取数据
    transaction_data = st.session_state.transaction_data
    user_behavior_data = st.session_state.user_behavior_data
    segmented_data = st.session_state.segmented_data
    transaction_features = st.session_state.transaction_features
    
    # 1. 构建用户行为图
    connection_strength, user_index = build_user_behavior_graph(user_behavior_data, transaction_data)
    
    # 2. 生成行为特征分布
    behavior_distributions = generate_behavior_feature_distribution(
        connection_strength, user_index, transaction_features, user_behavior_data
    )
    
    # 3. 计算时间偏移
    transaction_sequences = {}
    for window_id in transaction_features['window_id'].unique():
        window_data = segmented_data[segmented_data['window_id'] == window_id]
        transaction_sequences[window_id] = window_data['amount'].values
    
    time_offsets = calculate_time_offset(transaction_sequences)
    
    # 4. 计算行为一致性
    transaction_distributions = {wid: vals for wid, vals in transaction_sequences.items()}
    consistency_scores = calculate_behavior_consistency(
        transaction_distributions, behavior_distributions, time_offsets
    )
    
    # 5. 检测异常特征
    anomaly_features = detect_anomaly_features(transaction_features)
    
    # 6. 生成动态风险值
    dynamic_risk_values, core_anomaly_dims = generate_dynamic_risk_value(
        anomaly_features, transaction_features, behavior_distributions, consistency_scores
    )
    
    # 7. 生成欺诈标记
    fraud_labels = generate_fraud_labels(dynamic_risk_values, transaction_features, segmented_data)
    
    # 保存结果到session state
    st.session_state.consistency_scores = consistency_scores
    st.session_state.anomaly_features = anomaly_features
    st.session_state.dynamic_risk_values = dynamic_risk_values
    st.session_state.fraud_labels = fraud_labels

def detect_anomaly_features(transaction_features):
    """检测异常特征"""
    anomaly_features = {}
    
    for window_id in transaction_features['window_id'].unique():
        window_data = transaction_features[transaction_features['window_id'] == window_id].iloc[0]
        anomalies = []
        
        # 检测异常特征
        if window_data['fraud_ratio'] > 0.05:  # 欺诈率超过5%
            anomalies.append('fraud_ratio')
        if window_data['avg_amount'] > 5000:  # 平均金额过大
            anomalies.append('avg_amount')
        if window_data['max_amount'] > 50000:  # 最大金额异常
            anomalies.append('max_amount')
        if window_data['total_transactions'] < 10:  # 交易数量过少
            anomalies.append('total_transactions')
            
        anomaly_features[window_id] = anomalies
    
    return anomaly_features

def generate_fraud_labels(dynamic_risk_values, transaction_features, segmented_data):
    """生成欺诈标记结果 - 按用户聚合"""
    fraud_labels = []
    
    # 按用户聚合风险数据
    user_risk_data = {}
    
    for window_id in sorted(dynamic_risk_values.keys()):
        window_data = segmented_data[segmented_data['window_id'] == window_id]
        
        # 遍历该窗口的所有用户
        for user_id in window_data['user_id'].unique():
            user_window_data = window_data[window_data['user_id'] == user_id]
            
            if user_id not in user_risk_data:
                user_risk_data[user_id] = {
                    'risk_values': [],
                    'transaction_count': 0,
                    'fraud_count': 0,
                    'windows': []
                }
            
            user_risk_data[user_id]['risk_values'].append(dynamic_risk_values[window_id])
            user_risk_data[user_id]['transaction_count'] += len(user_window_data)
            user_risk_data[user_id]['fraud_count'] += user_window_data['is_fraud'].sum()
            user_risk_data[user_id]['windows'].append(window_id)
    
    # 为每个用户生成标记结果
    for user_id, data in user_risk_data.items():
        # 计算用户的平均风险值
        avg_risk_value = np.mean(data['risk_values'])
        max_risk_value = np.max(data['risk_values'])
        
        # 根据最高风险值判定（更严格）
        if max_risk_value > 0.7:
            status = "欺诈"
            color = "🔴"
        elif max_risk_value > 0.5:
            status = "欺诈"
            color = "🟡"
        else:
            status = "正常"
            color = "🟢"
        
        fraud_labels.append({
            'user_id': user_id,
            'risk_value': avg_risk_value,
            'max_risk_value': max_risk_value,
            'status': status,
            'color': color,
            'transaction_count': data['transaction_count'],
            'fraud_count': data['fraud_count'],
            'window_count': len(data['windows'])
        })
    
    return pd.DataFrame(fraud_labels).sort_values('max_risk_value', ascending=False)

def show_fraud_detection_results():
    """显示欺诈检测结果"""
    st.markdown("### 🎯 欺诈检测结果")
    
    fraud_labels = st.session_state.fraud_labels
    
    # 统计指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        fraud_count = len(fraud_labels[fraud_labels['status'] == '欺诈'])
        st.metric("欺诈行为", fraud_count, delta=f"{fraud_count/len(fraud_labels)*100:.1f}%", delta_color="inverse")
    with col2:
        normal_count = len(fraud_labels[fraud_labels['status'] == '正常'])
        st.metric("正常行为", normal_count, delta=f"{normal_count/len(fraud_labels)*100:.1f}%", delta_color="normal")
    with col3:
        total_windows = len(fraud_labels)
        st.metric("总时间窗口", total_windows)
    with col4:
        total_fraud = fraud_labels['fraud_count'].sum()
        st.metric("异常交易数", f"{total_fraud}")
    
    # 显示检测结果表格
    st.markdown("#### 用户行为标记结果")
    display_df = fraud_labels.copy()
    display_df['risk_value'] = display_df['risk_value'].apply(lambda x: f"{x:.3f}")
    display_df['max_risk_value'] = display_df['max_risk_value'].apply(lambda x: f"{x:.3f}")
    display_df = display_df[['color', 'user_id', 'status', 'max_risk_value', 'risk_value', 'transaction_count', 'fraud_count', 'window_count']]
    display_df.columns = ['状态', '用户ID', '行为标记', '最高风险值', '平均风险值', '交易数', '异常数', '窗口数']
    st.dataframe(display_df, use_container_width=True, height=400)

def show_anomaly_distribution():
    """显示异常特征分布"""
    st.markdown("### 📊 各时间窗口异常特征分布")
    
    anomaly_features = st.session_state.anomaly_features
    
    # 创建热力图
    fig = create_anomaly_heatmap(anomaly_features)
    st.plotly_chart(fig, use_container_width=True)

def create_anomaly_heatmap(anomaly_features):
    """创建异常特征热力图"""
    window_ids = sorted(anomaly_features.keys())
    all_features = set()
    for features in anomaly_features.values():
        all_features.update(features)
    all_features = sorted(list(all_features))
    
    if not all_features:
        all_features = ['无异常']
    
    # 创建特征矩阵
    matrix = []
    for window_id in window_ids:
        row = [1 if f in anomaly_features.get(window_id, []) else 0 for f in all_features]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=all_features,
        y=window_ids,
        colorscale='Reds',
        showscale=True,
        colorbar=dict(title='异常状态')
    ))
    
    fig.update_layout(
        title='异常特征分布热力图',
        xaxis_title='特征维度',
        yaxis_title='时间窗口ID',
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=500
    )
    
    return fig

def show_risk_trend():
    """显示风险值趋势"""
    st.markdown("### 📈 时间窗口动态欺诈风险值趋势")
    
    dynamic_risk_values = st.session_state.dynamic_risk_values
    
    fig = create_risk_trend_chart(dynamic_risk_values)
    st.plotly_chart(fig, use_container_width=True)

def create_risk_trend_chart(risk_values):
    """创建风险趋势图"""
    window_ids = sorted(risk_values.keys())
    risks = [risk_values[wid] for wid in window_ids]
    
    fig = go.Figure()
    
    # 添加风险曲线
    fig.add_trace(go.Scatter(
        x=window_ids,
        y=risks,
        mode='lines+markers',
        name='风险值',
        line=dict(color='#E74C3C', width=3),
        marker=dict(size=8, symbol='circle'),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.2)',
        hovertemplate='窗口 %{x}<br>风险值: %{y:.3f}<extra></extra>'
    ))
    
    # 添加风险阈值线
    fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                  annotation_text="高风险阈值", annotation_position="right")
    fig.add_hline(y=0.5, line_dash="dot", line_color="orange", 
                  annotation_text="中风险阈值", annotation_position="right")
    
    fig.update_layout(
        title='动态欺诈风险值趋势',
        xaxis_title='时间窗口ID',
        yaxis_title='风险值',
        plot_bgcolor='rgba(240,240,240,0.5)',
        yaxis=dict(range=[0, 1.1]),
        height=450,
        hovermode='x unified'
    )
    
    return fig

def show_transaction_analysis():
    """显示交易类型分析"""
    st.markdown("### 💰 交易类型分布与欺诈关联图")
    
    transaction_data = st.session_state.transaction_data
    
    fig = create_transaction_fraud_chart(transaction_data)
    st.plotly_chart(fig, use_container_width=True)

def create_transaction_fraud_chart(transaction_data):
    """创建交易类型欺诈关联图"""
    # 按交易类型统计
    trans_stats = transaction_data.groupby('transaction_type').agg({
        'is_fraud': ['sum', 'mean', 'count']
    }).reset_index()
    trans_stats.columns = ['transaction_type', 'fraud_count', 'fraud_rate', 'total_count']
    trans_stats = trans_stats.sort_values('fraud_rate', ascending=False)
    
    # 创建双轴图表
    fig = go.Figure()
    
    # 柱状图：交易数量
    fig.add_trace(go.Bar(
        x=trans_stats['transaction_type'],
        y=trans_stats['total_count'],
        name='交易数量',
        marker=dict(color='lightblue'),
        yaxis='y',
        hovertemplate='%{x}<br>交易数: %{y}<extra></extra>'
    ))
    
    # 折线图：欺诈率
    fig.add_trace(go.Scatter(
        x=trans_stats['transaction_type'],
        y=trans_stats['fraud_rate'] * 100,
        name='欺诈率 (%)',
        mode='lines+markers',
        line=dict(color='red', width=3),
        marker=dict(size=10, symbol='diamond'),
        yaxis='y2',
        hovertemplate='%{x}<br>欺诈率: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='交易类型分布与欺诈关联分析',
        xaxis_title='交易类型',
        yaxis=dict(title='交易数量', side='left'),
        yaxis2=dict(title='欺诈率 (%)', side='right', overlaying='y'),
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=450,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig

