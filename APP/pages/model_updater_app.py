import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pages.model_updater import FraudDetectionModel
import time
from datetime import datetime

def model_updater_app():
    st.title("模型更新与优化")
    
    # 检查是否有数据可用
    if 'user_data_generated' not in st.session_state or not st.session_state.user_data_generated:
        st.warning("⚠️ 请先在【用户数据】页面查看数据")
        return
    
    if 'analysis_completed' not in st.session_state or not st.session_state.analysis_completed:
        st.warning("⚠️ 请先在【风险分析评估】页面查看分析结果")
        return
    
    # 初始化模型历史
    if 'model_history' not in st.session_state:
        st.session_state.model_history = []
    
    # 模型训练部分
    st.markdown("### 模型训练与性能监控")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("训练/更新模型", type="primary", use_container_width=True):
            with st.spinner("正在训练模型..."):
                accuracy = train_model()
                if accuracy is not None:
                    st.success(f"✅ 模型训练完成！准确率: {accuracy:.2%}")
                    st.session_state.model_trained = True
                    st.rerun()
    
    with col2:
        if st.session_state.get('model_trained', False):
            st.metric("当前准确率", f"{st.session_state.model_history[-1]['accuracy']:.2%}" if st.session_state.model_history else "N/A")
    
    with col3:
        if st.session_state.model_history:
            st.metric("训练次数", len(st.session_state.model_history))
    
    # 显示模型性能变化图表
    if st.session_state.model_history:
        st.markdown("### 模型检测准确率变化")
        fig = create_model_performance_chart(st.session_state.model_history)
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示训练历史记录
        st.markdown("### 训练历史记录")
        history_df = pd.DataFrame(st.session_state.model_history)
        st.dataframe(history_df, use_container_width=True, height=300)
    else:
        st.info("暂无模型训练记录，请点击上方按钮开始训练")

def train_model():
    """训练欺诈检测模型"""
    try:
        # 获取数据
        transaction_features = st.session_state.transaction_features
        dynamic_risk_values = st.session_state.dynamic_risk_values
        consistency_scores = st.session_state.consistency_scores
        
        # 准备训练数据
        features = []
        labels = []
        window_ids = sorted(transaction_features["window_id"].unique())
        
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
                    consistency_scores.get(window_id, 0),
                    dynamic_risk_values.get(window_id, 0)
                ]
                # 标签：是否为高风险窗口
                label = 1 if dynamic_risk_values.get(window_id, 0) > 0.5 else 0
                features.append(feature)
                labels.append(label)
        
        # 初始化并更新模型
        if 'fraud_model' not in st.session_state:
            st.session_state.fraud_model = FraudDetectionModel()
        
        model = st.session_state.fraud_model
        model.update_parameters(dynamic_risk_values)
        
        # 训练模型
        X = np.array(features)
        y = np.array(labels)
        accuracy = model.train(X, y)
        
        # 保存训练记录
        st.session_state.model_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': accuracy,
            'samples': len(X),
            'fraud_rate': y.mean(),
            'n_estimators': model.params['n_estimators'],
            'max_depth': model.params['max_depth']
        })
        
        return accuracy
        
    except Exception as e:
        st.error(f"❌ 模型训练失败: {str(e)}")
        return None

def create_model_performance_chart(history):
    """创建模型性能变化图表"""
    iterations = list(range(1, len(history) + 1))
    accuracies = [h['accuracy'] for h in history]
    
    fig = go.Figure()
    
    # 添加准确率曲线
    fig.add_trace(go.Scatter(
        x=iterations,
        y=accuracies,
        mode='lines+markers',
        name='准确率',
        line=dict(color='#2E86DE', width=3),
        marker=dict(size=10, symbol='circle'),
        hovertemplate='训练次数: %{x}<br>准确率: %{y:.2%}<extra></extra>'
    ))
    
    # 添加目标线
    fig.add_hline(y=0.95, line_dash="dash", line_color="green", 
                  annotation_text="目标准确率 (95%)", annotation_position="right")
    
    fig.update_layout(
        title='模型检测准确率变化趋势',
        xaxis_title='训练次数',
        yaxis_title='准确率',
        plot_bgcolor='rgba(240,240,240,0.5)',
        yaxis=dict(
            tickformat='.0%',
            range=[0.5, 1.0]
        ),
        height=450,
        hovermode='x unified'
    )
    
    return fig