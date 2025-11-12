# 用户数据页面
import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import time
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx

data_title = ['金融消费交易记录', '用户行为', '用户行为交互网络', '行为一致性分析']

def tit_button(index):
    """按钮点击回调函数"""
    st.session_state.index = index
    st.session_state.info = data_title[index]

def users_data_app():
    st.title("用户数据采集与分析")
    
    # 初始化session state
    if 'user_data_generated' not in st.session_state:
        st.session_state.user_data_generated = False
        st.session_state.index = 0
        st.session_state.info = data_title[0]
        st.session_state.last_update_time = None
    
    # 自动生成和更新数据
    current_time = datetime.now()
    
    # 首次加载或超过1分钟自动更新
    should_update = False
    if not st.session_state.user_data_generated:
        should_update = True
    elif st.session_state.last_update_time is not None:
        time_diff = (current_time - st.session_state.last_update_time).total_seconds()
        if time_diff >= 60:  # 60秒 = 1分钟
            should_update = True
    
    if should_update:
        with st.spinner("正在采集数据..."):
            # 生成数据
            transaction_data = generate_financial_transaction_data(num_users=100, num_days=7, transactions_per_day=500)
            user_behavior_data = generate_user_behavior_data(transaction_data)
            segmented_data, windows = segment_data_by_time_window(transaction_data, window_size="6H")
            transaction_features = extract_transaction_features(segmented_data)
            
            # 保存到session state
            st.session_state.transaction_data = transaction_data
            st.session_state.user_behavior_data = user_behavior_data
            st.session_state.segmented_data = segmented_data
            st.session_state.windows = windows
            st.session_state.transaction_features = transaction_features
            st.session_state.user_data_generated = True
            st.session_state.last_update_time = current_time
            
            # 清除之前的分析结果，需要重新分析
            if 'analysis_completed' in st.session_state:
                st.session_state.analysis_completed = False
    
    # 显示最后更新时间
    if st.session_state.last_update_time:
        time_since_update = (current_time - st.session_state.last_update_time).total_seconds()
        next_update_in = max(0, 60 - time_since_update)
        st.info(f"数据自动更新 | 最后更新: {st.session_state.last_update_time.strftime('%H:%M:%S')} | 下次更新: {int(next_update_in)}秒后")
        
        # 自动刷新
        if next_update_in <= 0:
            time.sleep(1)
            st.rerun()
    
    tp = lambda x: 'primary' if st.session_state.index == x else 'secondary'
    
    col01, col02 = st.columns([1, 5])   # 左侧按钮列，右侧内容列
    
    with col01:
        # st.markdown("### 📋 数据视图")
        with st.container(height=600, border=True):
            for ind, tit in enumerate(data_title):
                if st.button(label=tit, key=f'tit_{ind}', use_container_width=True, 
                           on_click=tit_button, args=(ind,), type=tp(ind)):
                    pass
            
            st.divider()
            if st.button("🔄 立即刷新", use_container_width=True):
                st.session_state.user_data_generated = False
                st.session_state.last_update_time = None
                st.rerun()
    
    with col02:
        with st.container(border=True, height=600):
            if st.session_state.index == 0:
                show_transaction_data()
            elif st.session_state.index == 1:
                show_user_behavior_data()
            elif st.session_state.index == 2:
                show_behavior_network()
            elif st.session_state.index == 3:
                show_consistency_analysis()

def show_transaction_data():
    """显示金融消费交易数据"""
    st.markdown("### 金融消费交易数据")
    
    transaction_data = st.session_state.transaction_data
    
    # 数据概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总交易数", f"{len(transaction_data):,}")
    with col2:
        st.metric("欺诈交易数", f"{transaction_data['is_fraud'].sum():,}")
    with col3:
        st.metric("欺诈率", f"{transaction_data['is_fraud'].mean()*100:.2f}%")
    with col4:
        st.metric("总金额", f"¥{transaction_data['amount'].sum():,.0f}")
    
    # 数据表格
    st.dataframe(transaction_data.head(100), use_container_width=True, height=400)

def show_user_behavior_data():
    """显示多维用户行为数据"""
    st.markdown("### 多维用户行为数据")
    
    user_behavior_data = st.session_state.user_behavior_data
    
    # 数据概览
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("用户总数", len(user_behavior_data))
    with col2:
        st.metric("平均信用分", f"{user_behavior_data['credit_score'].mean():.0f}")
    with col3:
        st.metric("平均年龄", f"{user_behavior_data['age'].mean():.0f}岁")
    with col4:
        st.metric("平均交易额", f"¥{user_behavior_data['avg_trans_amount'].mean():,.0f}")
    
    # 数据表格
    st.dataframe(user_behavior_data, use_container_width=True, height=400)

def show_behavior_network():
    """显示用户行为交互网络图"""
    st.markdown("### 用户行为交互网络图")
    
    with st.spinner("正在生成网络图..."):
        transaction_data = st.session_state.transaction_data
        user_behavior_data = st.session_state.user_behavior_data
        
        # 构建网络图
        fig = create_behavior_network_plotly(transaction_data, user_behavior_data)
        st.plotly_chart(fig, use_container_width=True)

def show_consistency_analysis():
    """显示行为一致性分析"""
    st.markdown("### 各时间窗口行为一致性指标")
    
    with st.spinner("正在计算行为一致性..."):
        segmented_data = st.session_state.segmented_data
        transaction_features = st.session_state.transaction_features
        
        # 计算一致性分数（简化版）
        consistency_scores = {}
        for window_id in transaction_features['window_id'].unique():
            window_data = segmented_data[segmented_data['window_id'] == window_id]
            if len(window_data) > 0:
                # 基于交易时间分布的一致性
                fraud_ratio = window_data['is_fraud'].mean()
                amount_std = window_data['amount'].std()
                normalized_std = min(amount_std / 10000, 1.0)
                consistency = 1 - fraud_ratio - normalized_std * 0.3
                consistency_scores[window_id] = max(0, min(1, consistency))
        
        # 保存到session state
        st.session_state.consistency_scores = consistency_scores
        
        # 绘制图表
        fig = create_consistency_chart(consistency_scores)
        st.plotly_chart(fig, use_container_width=True)

def create_behavior_network_plotly(transaction_data, user_behavior_data):
    """使用Plotly创建用户行为交互网络图"""
    # 选择部分用户（前30个）以提高可视化效果
    users = user_behavior_data['user_id'].head(30).tolist()
    
    # 创建网络图
    G = nx.Graph()
    
    # 添加节点
    for user in users:
        user_info = user_behavior_data[user_behavior_data['user_id'] == user].iloc[0]
        G.add_node(user, 
                  credit_score=user_info['credit_score'],
                  age=user_info['age'])
    
    # 添加边（基于转账关系）
    transfers = transaction_data[transaction_data['transaction_type'] == '转账']
    for _, trans in transfers.head(100).iterrows():
        if trans['user_id'] in users:
            # 随机选择一个接收者
            recipient = random.choice([u for u in users if u != trans['user_id']])
            if G.has_edge(trans['user_id'], recipient):
                G[trans['user_id']][recipient]['weight'] += trans['amount']
            else:
                G.add_edge(trans['user_id'], recipient, weight=trans['amount'])
    
    # 使用spring layout
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # 创建边的traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(x=[x0, x1, None], y=[y0, y1, None],
                      mode='lines',
                      line=dict(width=0.5, color='#888'),
                      hoverinfo='none',
                      showlegend=False)
        )
    
    # 创建节点trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        user_info = user_behavior_data[user_behavior_data['user_id'] == node].iloc[0]
        node_text.append(f"用户: {node}<br>信用分: {user_info['credit_score']}<br>年龄: {user_info['age']}")
        node_color.append(user_info['credit_score'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_color,
            size=15,
            colorbar=dict(
                thickness=15,
                title=dict(text='信用分'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )
    
    # 创建图形
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title='用户行为交互网络',
        title_font_size=16,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=500
    )
    
    return fig

def create_consistency_chart(consistency_scores):
    """创建行为一致性图表"""
    window_ids = sorted(consistency_scores.keys())
    scores = [consistency_scores[wid] for wid in window_ids]
    
    fig = go.Figure()
    
    # 添加柱状图
    fig.add_trace(go.Bar(
        x=window_ids,
        y=scores,
        marker=dict(
            color=scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='一致性分数')
        ),
        text=[f'{s:.2f}' for s in scores],
        textposition='outside',
        hovertemplate='窗口 %{x}<br>一致性: %{y:.3f}<extra></extra>'
    ))
    
    # 添加阈值线
    fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                  annotation_text="一致性阈值", annotation_position="right")
    
    fig.update_layout(
        title='各时间窗口行为一致性指标',
        xaxis_title='时间窗口ID',
        yaxis_title='行为一致性指标',
        plot_bgcolor='rgba(240,240,240,0.5)',
        height=450,
        showlegend=False
    )
    
    return fig

def generate_financial_transaction_data(num_users=1000, num_days=30, transactions_per_day=10000):
    """金融消费交易数据"""
    # 生成时间序列
    start_date = datetime.now() - timedelta(days=num_days)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # 用户ID
    user_ids = [f"USER{i:04d}" for i in range(num_users)]
    
    # 交易类型
    transaction_types = ["消费", "转账", "提现", "还款", "充值"]
    
    # 交易金额范围（元）
    amount_ranges = {
        "消费": (10, 5000),
        "转账": (100, 100000),
        "提现": (500, 50000),
        "还款": (100, 50000),
        "充值": (100, 100000)
    }
    
    # 位置信息
    locations = ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京", "重庆", "西安"]
    
    data = []
    total = num_days * transactions_per_day
    count = 0
    
    for date in dates:
        for _ in range(transactions_per_day):
            user_id = random.choice(user_ids)
            trans_type = random.choice(transaction_types)
            min_amt, max_amt = amount_ranges[trans_type]
            amount = round(random.uniform(min_amt, max_amt), 2)
            
            is_fraud = 1 if random.random() < 0.01 else 0
            if is_fraud:
                # 欺诈交易通常金额较大或时间异常
                amount = round(amount * random.uniform(2, 10), 2)
                hour = random.choice([0, 1, 2, 3, 4, 22, 23])  # 凌晨异常时间
            else:
                hour = random.choice(range(8, 22))  # 正常交易时间
            
            trans_time = date.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))
            location = random.choice(locations)
            
            data.append({
                "user_id": user_id,
                "transaction_time": trans_time,
                "transaction_type": trans_type,
                "amount": amount,
                "location": location,
                "is_fraud": is_fraud
            })
            
            count += 1
            if count % 1000 == 0:
                time.sleep(0.01)
    
    df = pd.DataFrame(data)
    df = df.sort_values(["user_id", "transaction_time"])
    return df

def generate_user_behavior_data(transaction_data):
    """多维用户行为数据"""
    user_ids = transaction_data["user_id"].unique()
    total_users = len(user_ids)
    
    # 行为特征
    behaviors = []
    
    for i, user_id in enumerate(user_ids):
        # 用户基本信息
        age = random.randint(18, 65)
        gender = random.choice(["男", "女"])
        income_level = random.choice(["低收入", "中等收入", "高收入"])
        credit_score = random.randint(300, 850)
        
        # 设备使用习惯
        devices = random.choice(["手机", "电脑", "平板", "多设备"])
        login_frequency = random.choice(["高频", "中频", "低频"])
        
        # 交易习惯
        user_trans = transaction_data[transaction_data["user_id"] == user_id]
        avg_trans_amount = user_trans["amount"].mean()
        preferred_trans_type = user_trans["transaction_type"].mode().values[0] if not user_trans.empty else "消费"
        
        behaviors.append({
            "user_id": user_id,
            "age": age,
            "gender": gender,
            "income_level": income_level,
            "credit_score": credit_score,
            "devices": devices,
            "login_frequency": login_frequency,
            "avg_trans_amount": round(avg_trans_amount, 2) if not user_trans.empty else 0,
            "preferred_trans_type": preferred_trans_type
        })
        
        if i % 100 == 0:
            time.sleep(0.01)
    
    df = pd.DataFrame(behaviors)
    return df

def segment_data_by_time_window(transaction_data, window_size="6h"):
    """根据预设的时间窗口对金融消费交易数据进行分段处理"""
    # 确保数据按时间排序
    transaction_data = transaction_data.sort_values("transaction_time").copy()
    
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
        
        # 添加延迟以控制进度
        if i % 10 == 0:
            time.sleep(0.05)
    
    return transaction_data, windows

def extract_transaction_features(segmented_data):
    """提取交易特征"""
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
        # 更新进度
        if i % 10 == 0:
            time.sleep(0.05)
    
    features_df = pd.DataFrame(window_features)
    return features_df