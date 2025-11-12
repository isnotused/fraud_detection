### 系统管理
import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

def system_management():
    st.title("⚙️ 系统管理")
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["数据管理", "模型配置", "系统监控", "日志管理"])
    
    with tab1:
        show_data_management()
    
    with tab2:
        show_model_configuration()
    
    with tab3:
        show_system_monitoring()
    
    with tab4:
        show_log_management()

def show_data_management():
    """数据管理"""
    st.markdown("### 数据管理")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_data = 0
        if 'transaction_data' in st.session_state:
            total_data = len(st.session_state.transaction_data)
        st.metric("总交易记录", f"{total_data:,}")
    
    with col2:
        total_users = 0
        if 'user_behavior_data' in st.session_state:
            total_users = len(st.session_state.user_behavior_data)
        st.metric("总用户数", f"{total_users}")
    
    with col3:
        total_windows = 0
        if 'transaction_features' in st.session_state:
            total_windows = len(st.session_state.transaction_features)
        st.metric("时间窗口数", f"{total_windows}")
    
    with col4:
        fraud_detected = 0
        if 'fraud_labels' in st.session_state:
            fraud_detected = st.session_state.fraud_labels['fraud_count'].sum()
        st.metric("检测到欺诈", f"{fraud_detected}")
    
    st.divider()
    
    # 数据概览
    st.markdown("#### 数据集概览")
    
    if 'user_data_generated' in st.session_state and st.session_state.user_data_generated:
        data_info = []
        
        if 'transaction_data' in st.session_state:
            df = st.session_state.transaction_data
            data_info.append({
                '数据集': '交易数据',
                '记录数': len(df),
                '字段数': len(df.columns),
                '内存占用': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                '最后更新': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if 'user_behavior_data' in st.session_state:
            df = st.session_state.user_behavior_data
            data_info.append({
                '数据集': '用户行为数据',
                '记录数': len(df),
                '字段数': len(df.columns),
                '内存占用': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
                '最后更新': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        if data_info:
            st.dataframe(pd.DataFrame(data_info), use_container_width=True, hide_index=True)
    else:
        st.info("暂无数据，请先生成数据")
    
    st.divider()
    
    # 数据操作
    st.markdown("#### 数据操作")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 刷新数据", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("导出数据", use_container_width=True):
            st.info("数据导出功能开发中...")
    
    with col3:
        if st.button("🗑️ 清除数据", use_container_width=True, type="secondary"):
            if 'user_data_generated' in st.session_state:
                st.session_state.user_data_generated = False
                st.session_state.analysis_completed = False
                st.success("数据已清除")
                st.rerun()

def show_model_configuration():
    """模型配置"""
    st.markdown("### 模型配置")
    
    # 当前模型参数
    st.markdown("#### 当前模型参数")
    
    if 'fraud_model' in st.session_state:
        model = st.session_state.fraud_model
        params = model.params
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("决策树数量", params.get('n_estimators', 50))
            st.metric("最大深度", params.get('max_depth', 10))
        
        with col2:
            st.metric("最小分裂样本", params.get('min_samples_split', 2))
            st.metric("最小叶子样本", params.get('min_samples_leaf', 1))
    else:
        st.info("模型尚未初始化")
    
    st.divider()
    
    # 参数调整
    st.markdown("#### 参数调整")
    
    with st.form("model_params_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("决策树数量", 10, 200, 50, 10)
            max_depth = st.slider("最大深度", 3, 30, 10, 1)
        
        with col2:
            min_samples_split = st.slider("最小分裂样本", 2, 20, 2, 1)
            min_samples_leaf = st.slider("最小叶子样本", 1, 20, 1, 1)
        
        submitted = st.form_submit_button("应用配置", use_container_width=True, type="primary")
        
        if submitted:
            if 'fraud_model' in st.session_state:
                st.session_state.fraud_model.params.update({
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf
                })
                st.session_state.fraud_model.model.set_params(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                st.success("✅ 参数配置已更新")
            else:
                st.warning("请先训练模型")

def show_system_monitoring():
    """系统监控"""
    st.markdown("### 系统监控")
    
    # 系统状态
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status = "🟢 运行中" if 'user_data_generated' in st.session_state else "🔴 未启动"
        st.metric("系统状态", status)
    
    with col2:
        model_status = "✅ 已训练" if st.session_state.get('model_trained', False) else "⚠️ 未训练"
        st.metric("模型状态", model_status)
    
    with col3:
        analysis_status = "✅ 已完成" if st.session_state.get('analysis_completed', False) else "⚠️ 未完成"
        st.metric("分析状态", analysis_status)
    
    with col4:
        train_count = len(st.session_state.get('model_history', []))
        st.metric("训练次数", train_count)
    
    st.divider()
    
    # 性能指标
    st.markdown("#### 性能指标")
    
    if st.session_state.get('model_history'):
        history = st.session_state.model_history
        latest = history[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("当前准确率", f"{latest['accuracy']:.2%}")
        
        with col2:
            if len(history) > 1:
                prev_accuracy = history[-2]['accuracy']
                delta = latest['accuracy'] - prev_accuracy
                st.metric("准确率变化", f"{delta:+.2%}")
            else:
                st.metric("准确率变化", "N/A")
        
        with col3:
            st.metric("训练样本数", latest['samples'])
        
        # 性能趋势图
        if len(history) > 1:
            st.markdown("#### 性能趋势")
            fig = create_performance_trend(history)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无性能数据")

def create_performance_trend(history):
    """创建性能趋势图"""
    iterations = list(range(1, len(history) + 1))
    accuracies = [h['accuracy'] for h in history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=iterations,
        y=accuracies,
        mode='lines+markers',
        name='准确率',
        line=dict(color='#3498DB', width=2),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='模型性能趋势',
        xaxis_title='训练次数',
        yaxis_title='准确率',
        plot_bgcolor='rgba(240,240,240,0.5)',
        yaxis=dict(tickformat='.0%'),
        height=300
    )
    
    return fig

def show_log_management():
    """日志管理"""
    st.markdown("### 日志管理")
    
    # 操作日志
    st.markdown("#### 系统操作日志")
    
    # 生成模拟日志
    logs = []
    
    if 'user_data_generated' in st.session_state and st.session_state.user_data_generated:
        logs.append({
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '操作': '数据生成',
            '状态': '✅ 成功',
            '详情': '生成用户交易数据和行为数据'
        })
    
    if 'analysis_completed' in st.session_state and st.session_state.analysis_completed:
        logs.append({
            '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '操作': '风险分析',
            '状态': '✅ 成功',
            '详情': '完成异常检测和风险评估'
        })
    
    if 'model_history' in st.session_state:
        for i, record in enumerate(st.session_state.model_history):
            logs.append({
                '时间': record['timestamp'],
                '操作': '模型训练',
                '状态': '✅ 成功',
                '详情': f"准确率: {record['accuracy']:.2%}"
            })
    
    if logs:
        log_df = pd.DataFrame(logs)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
    else:
        st.info("暂无操作日志")
    
    st.divider()
    
    # 日志操作
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 刷新日志", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("导出日志", use_container_width=True):
            st.info("日志导出功能开发中...")
    
    with col3:
        if st.button("🗑️ 清除日志", use_container_width=True, type="secondary"):
            st.warning("确认清除所有日志？")
