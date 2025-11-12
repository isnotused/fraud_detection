
# 🏦 基于大数据的金融领域消费欺诈检测系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.51.0-FF4B4B.svg)](https://streamlit.io/)

## 🎯 项目简介

这是一个功能完整的金融欺诈检测系统，实现了从数据采集、风险分析、模型训练到系统管理的全流程智能化检测。

### ✨ 核心特性

- 🔍 **智能欺诈检测**：基于机器学习的多维度欺诈识别
- 📊 **实时数据分析**：动态时间窗口分析和风险评估
- 🤖 **自适应学习**：模型参数根据风险值自动调整
- 📈 **美观可视化**：Plotly交互式图表，直观展示分析结果
- ⚡ **高性能处理**：高效处理大规模交易数据

## 🚀 快速开始

### 启动应用

```bash
cd /Users/fuwei/fraud_detection
uv run streamlit run APP/head.py
```

应用将在浏览器中自动打开：**http://localhost:8502**

### 使用流程

1. **生成数据**：进入【用户数据】→ 点击"生成模拟数据"
2. **风险分析**：进入【风险分析评估】→ 点击"开始风险分析"
3. **模型训练**：进入【模型更新】→ 点击"训练/更新模型"
4. **系统管理**：进入【系统管理】→ 查看数据和监控

## 📋 系统功能

### 1. 用户数据采集 📊
- 金融消费交易数据
- 多维用户行为数据
- 用户行为交互网络
- 行为一致性分析

### 2. 风险分析评估 🔍
- 🔴🟡🟢 欺诈风险等级标记
- 异常特征分布热力图
- 动态风险值趋势图
- 交易类型欺诈关联图

### 3. 模型更新优化 🤖
- RandomForest欺诈检测模型
- 模型准确率变化监控
- 训练历史记录

### 4. 系统管理 ⚙️
- 数据管理
- 模型配置
- 系统监控
- 日志管理

## 📁 项目结构

```
APP/
├── head.py                    # 主入口
├── pages/                     # 页面模块
│   ├── users_data_app.py     # 用户数据
│   ├── data_analyzer_app.py  # 风险分析
│   ├── model_updater_app.py  # 模型更新
│   └── system_management.py  # 系统管理
├── 快速使用指南.md            # 快速指南
├── 系统使用说明.md            # 详细说明
└── 系统架构说明.md            # 技术文档
```

## 🔧 技术栈

- **Python 3.12** + **Streamlit 1.51.0**
- **Pandas** + **NumPy** + **Scikit-learn**
- **Plotly** + **NetworkX**

## 📝 详细文档

- [快速使用指南](快速使用指南.md)
- [系统使用说明](系统使用说明.md)
- [系统架构说明](系统架构说明.md)
- [项目完成总结](项目完成总结.md)

## ⚙️ Streamlit 配置

关闭默认多页导航栏：

```bash
# 方法1：启动时添加参数
streamlit run head.py --client.showSidebarNavigation false

# 方法2：修改 config.toml
showSidebarNavigation = false
```

---

**© 2025 金融欺诈检测系统 | 智能风控，守护安全**

