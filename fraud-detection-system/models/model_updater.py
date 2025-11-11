import numpy as np
import pandas as pd
from sklearn.base import clone

class ModelUpdater:
    """模型更新器，负责基于动态欺诈风险值更新检测模型参数"""
    
    def __init__(self, base_model, update_threshold=0.5):
        """初始化模型更新器
        
        参数:
            base_model: 基础模型
            update_threshold: 触发模型更新的风险值阈值
        """
        self.base_model = base_model
        self.update_threshold = update_threshold
        self.models = [clone(base_model)]  # 存储模型版本历史
        self.current_model_index = 0
        
    def get_current_model(self):
        """获取当前使用的模型
        
        返回:
            当前模型
        """
        return self.models[self.current_model_index]
        
    def calculate_temporary_parameters(self, model_parameters, risk_value):
        """计算当前检测模型的临时参数
        
        参数:
            model_parameters: 模型参数
            risk_value: 动态欺诈风险值
            
        返回:
            临时参数
        """
        # 将模型参数与风险值相乘得到临时参数
        return {param: value * risk_value for param, value in model_parameters.items()}
        
    def aggregate_parameters(self, temporary_params_list):
        """聚合多个时间窗口的临时参数
        
        参数:
            temporary_params_list: 临时参数列表
            
        返回:
            聚合后的参数
        """
        if not temporary_params_list:
            return self.get_current_model_parameters()
            
        # 初始化聚合参数字典
        aggregated_params = {}
        
        # 获取所有参数名称
        param_names = temporary_params_list[0].keys()
        
        # 对每个参数取平均值
        for param in param_names:
            param_values = [params[param] for params in temporary_params_list]
            aggregated_params[param] = np.mean(param_values)
            
        return aggregated_params
        
    def get_current_model_parameters(self):
        """获取当前模型的参数
        
        返回:
            模型参数字典
        """
        model = self.get_current_model()
        return model.get_params()
        
    def update_model_parameters(self, risk_values):
        """基于动态风险值更新模型参数
        
        参数:
            risk_values: 各时间窗口的动态风险值列表
            
        返回:
            更新后的模型
        """
        # 获取当前模型参数
        current_params = self.get_current_model_parameters()
        
        # 计算每个窗口的临时参数
        temporary_params_list = []
        for risk in risk_values:
            temp_params = self.calculate_temporary_parameters(current_params, risk)
            temporary_params_list.append(temp_params)
            
        # 聚合临时参数得到新参数
        new_params = self.aggregate_parameters(temporary_params_list)
        
        # 创建新模型并设置参数
        new_model = clone(self.base_model)
        new_model.set_params(** new_params)
        
        # 将新模型添加到模型历史并设为当前模型
        self.models.append(new_model)
        self.current_model_index = len(self.models) - 1
        
        return new_model
        
    def should_update(self, risk_values):
        """判断是否需要更新模型
        
        参数:
            risk_values: 风险值列表
            
        返回:
            是否需要更新的布尔值
        """
        if not risk_values:
            return False
            
        # 如果平均风险值超过阈值，则需要更新模型
        avg_risk = np.mean(risk_values)
        return avg_risk > self.update_threshold
        
    def retrain_with_new_data(self, X, y):
        """使用新数据重新训练模型
        
        参数:
            X: 新的特征数据
            y: 新的标签数据
            
        返回:
            重新训练后的模型
        """
        # 创建新模型并训练
        new_model = clone(self.base_model)
        new_model.fit(X, y)
        
        # 将新模型添加到模型历史并设为当前模型
        self.models.append(new_model)
        self.current_model_index = len(self.models) - 1
        
        return new_model
        
    def get_model_history(self):
        """获取模型历史记录
        
        返回:
            模型历史列表
        """
        return self.models
        
    def revert_to_previous_model(self):
        """回退到上一个模型版本
        
        返回:
            回退后的当前模型，如果已是第一个模型则返回自身
        """
        if self.current_model_index > 0:
            self.current_model_index -= 1
        return self.get_current_model()
