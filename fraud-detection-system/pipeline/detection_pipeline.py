import numpy as np
import pandas as pd
from datetime import datetime

from data.data_loader import DataLoader
from data.data_preprocessor import DataPreprocessor
from data.time_window_processor import TimeWindowProcessor
from features.feature_extractor import FeatureExtractor
from analysis.temporal_offset import TemporalOffsetAnalyzer
from analysis.behavior_consistency import BehaviorConsistencyAnalyzer
from analysis.anomaly_probability import AnomalyProbabilityCalculator
from analysis.risk_calculator import FraudRiskCalculator
from models.detection_model import FraudDetectionModel
from models.model_updater import ModelUpdater
from models.ensemble_classifier import EnsembleClassifier

class FraudDetectionPipeline:
    """欺诈检测流程管道，整合所有组件完成端到端的欺诈检测"""
    
    def __init__(self, config):
        """初始化欺诈检测流程管道
        
        参数:
            config: 配置字典
        """
        # 初始化配置
        self.config = config
        
        # 初始化数据处理组件
        self.data_loader = DataLoader(
            config['transaction_data_path'],
            config['user_behavior_data_path']
        )
        self.preprocessor = DataPreprocessor()
        self.window_processor = TimeWindowProcessor(
            window_size=config.get('window_size', 60),
            slide_step=config.get('slide_step', 30)
        )
        
        # 初始化特征提取组件
        self.feature_extractor = FeatureExtractor()
        
        # 初始化分析组件
        self.offset_analyzer = TemporalOffsetAnalyzer(
            max_offset=config.get('max_offset', 100),
            step=config.get('offset_step', 1)
        )
        self.consistency_analyzer = BehaviorConsistencyAnalyzer()
        self.anomaly_calculator = AnomalyProbabilityCalculator(
            anomaly_threshold=config.get('anomaly_threshold', 0.7)
        )
        self.risk_calculator = FraudRiskCalculator(
            core_weight=config.get('core_weight', 0.6),
            trajectory_weight=config.get('trajectory_weight', 0.4)
        )
        
        # 初始化模型组件
        self.base_model = FraudDetectionModel(
            model_type=config.get('model_type', 'rf'),
            random_state=config.get('random_state', 42)
        )
        self.model_updater = ModelUpdater(
            self.base_model,
            update_threshold=config.get('update_threshold', 0.5)
        )
        self.ensemble_classifier = EnsembleClassifier(
            base_model=self.base_model,
            n_estimators=config.get('ensemble_size', 5)
        )
        
        # 存储中间结果
        self.processed_data = {}
        self.results = {}
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        # 加载原始数据
        transaction_data, behavior_data = self.data_loader.load_all_data()
        
        # 清洗数据
        cleaned_transactions = self.preprocessor.clean_transaction_data(transaction_data)
        cleaned_behavior = self.preprocessor.clean_user_behavior_data(behavior_data)
        
        # 合并用户数据
        combined_data = self.preprocessor.combine_user_data(
            cleaned_transactions, 
            cleaned_behavior
        )
        
        # 存储预处理后的数据
        self.processed_data['raw_transactions'] = transaction_data
        self.processed_data['raw_behavior'] = behavior_data
        self.processed_data['cleaned_transactions'] = cleaned_transactions
        self.processed_data['cleaned_behavior'] = cleaned_behavior
        self.processed_data['combined_data'] = combined_data
        
        return combined_data
        
    def segment_data_by_window(self):
        """按时间窗口分割数据"""
        if 'combined_data' not in self.processed_data:
            self.load_and_preprocess_data()
            
        combined_data = self.processed_data['combined_data']
        windowed_data = {}
        
        # 为每个用户分割窗口数据
        for user_id, user_data in combined_data.items():
            user_window_data = self.window_processor.segment_user_data(user_data)
            if user_window_data:  # 只保留有数据的用户
                windowed_data[user_id] = user_window_data
                
        self.processed_data['windowed_data'] = windowed_data
        return windowed_data
        
    def extract_window_features(self):
        """提取每个时间窗口的特征"""
        if 'windowed_data' not in self.processed_data:
            self.segment_data_by_window()
            
        windowed_data = self.processed_data['windowed_data']
        feature_data = {}
        
        # 为每个用户提取窗口特征
        for user_id, user_window_data in windowed_data.items():
            user_features = self.feature_extractor.extract_user_features(user_window_data)
            
            # 组织特征数据
            feature_info = {
                'window_features': user_features,
                'transaction_distributions': {},
                'behavior_distributions': {},
                'offsets': {}
            }
            
            # 提取分布特征和计算偏移量
            window_indices = sorted(user_features.keys())
            for i, window_idx in enumerate(window_indices):
                window_data = user_window_data[window_idx]
                
                # 提取交易特征分布
                trans_features = self.feature_extractor.extract_transaction_features(
                    window_data['transactions']
                )
                if not trans_features.empty:
                    feature_info['transaction_distributions'][window_idx] = trans_features.values[0]
                    
                # 提取行为特征分布
                behav_features = self.feature_extractor.extract_behavior_features(
                    window_data['behavior']
                )
                if not behav_features.empty:
                    feature_info['behavior_distributions'][window_idx] = behav_features.values[0]
                
                # 计算与前一个窗口的偏移量
                if i > 0:
                    prev_window_idx = window_indices[i-1]
                    prev_trans = feature_info['transaction_distributions'].get(prev_window_idx, [])
                    curr_trans = feature_info['transaction_distributions'].get(window_idx, [])
                    
                    if len(prev_trans) > 0 and len(curr_trans) > 0:
                        offset = self.offset_analyzer.find_optimal_offset(prev_trans, curr_trans)
                        feature_info['offsets'][window_idx] = offset
                    else:
                        feature_info['offsets'][window_idx] = 0
                else:
                    feature_info['offsets'][window_idx] = 0
                    
            feature_data[user_id] = feature_info
            
        self.processed_data['feature_data'] = feature_data
        return feature_data
        
    def analyze_behavior_consistency(self):
        """分析用户行为一致性"""
        if 'feature_data' not in self.processed_data:
            self.extract_window_features()
            
        feature_data = self.processed_data['feature_data']
        consistency_data = {}
        
        # 为每个用户分析行为一致性
        for user_id, features in feature_data.items():
            window_indices = sorted(features['window_features'].keys())
            if len(window_indices) < 1:
                continue
                
            # 收集交易和行为分布
            trans_dists = []
            behav_dists = []
            offsets = []
            
            for window_idx in window_indices:
                trans_dists.append(features['transaction_distributions'].get(window_idx, []))
                behav_dists.append(features['behavior_distributions'].get(window_idx, []))
                offsets.append(features['offsets'].get(window_idx, 0))
                
            # 计算一致性分数
            consistency_scores = self.consistency_analyzer.calculate_consistency_sequence(
                trans_dists, behav_dists, offsets
            )
            
            # 计算一致性变化
            consistency_changes = self.consistency_analyzer.get_consistency_changes(consistency_scores)
            
            # 存储结果
            consistency_info = {
                'window_indices': window_indices,
                'consistency_scores': consistency_scores,
                'consistency_changes': consistency_changes
            }
            
            consistency_data[user_id] = consistency_info
            
        self.processed_data['consistency_data'] = consistency_data
        return consistency_data
        
    def calculate_anomaly_probabilities(self):
        """计算异常概率和识别异常特征"""
        if 'consistency_data' not in self.processed_data:
            self.analyze_behavior_consistency()
            
        feature_data = self.processed_data['feature_data']
        consistency_data = self.processed_data['consistency_data']
        anomaly_data = {}
        
        # 为每个用户计算异常概率
        for user_id, features in feature_data.items():
            if user_id not in consistency_data:
                continue
                
            consistency_info = consistency_data[user_id]
            window_indices = consistency_info['window_indices']
            consistency_scores = consistency_info['consistency_scores']
            consistency_changes = consistency_info['consistency_changes']
            
            if len(window_indices) < 1:
                continue
                
            # 准备窗口数据
            window_data_list = []
            for window_idx in window_indices:
                # 构建窗口数据结构
                window_feat = features['window_features'][window_idx]
                window_data = {
                    'features': window_feat.values if window_feat is not None and not window_feat.empty else np.array([]),
                    'distributions': features['transaction_distributions'].get(window_idx, []),
                    'behavior_features': features['behavior_distributions'].get(window_idx, [])
                }
                window_data_list.append(window_data)
                
            # 计算每个窗口的异常概率
            anomaly_probs = []
            anomaly_features = []
            
            for i in range(len(window_data_list)):
                curr_window_data = window_data_list[i]
                prev_window_data = window_data_list[i-1] if i > 0 else None
                
                # 计算异常概率
                probs = self.anomaly_calculator.calculate_window_anomaly_probabilities(
                    prev_window_data,
                    curr_window_data,
                    consistency_scores[i],
                    consistency_changes[i]
                )
                
                # 识别异常特征
                anomaly_feats = self.anomaly_calculator.identify_anomaly_features(probs)
                
                anomaly_probs.append(probs)
                anomaly_features.append(anomaly_feats)
                
            # 存储结果
            anomaly_info = {
                'window_indices': window_indices,
                'anomaly_probabilities': anomaly_probs,
                'anomaly_features': anomaly_features
            }
            
            anomaly_data[user_id] = anomaly_info
            
        self.processed_data['anomaly_data'] = anomaly_data
        return anomaly_data
        
    def calculate_risk_values(self):
        """计算动态欺诈风险值"""
        if 'anomaly_data' not in self.processed_data:
            self.calculate_anomaly_probabilities()
            
        feature_data = self.processed_data['feature_data']
        anomaly_data = self.processed_data['anomaly_data']
        risk_data = {}
        
        # 为每个用户计算风险值
        for user_id, anomaly_info in anomaly_data.items():
            if user_id not in feature_data:
                continue
                
            features = feature_data[user_id]
            window_indices = anomaly_info['window_indices']
            anomaly_features = anomaly_info['anomaly_features']
            
            if len(window_indices) < 1:
                continue
                
            # 准备窗口数据列表
            window_data_list = []
            for window_idx in window_indices:
                window_feat = features['window_features'][window_idx]
                window_data = {
                    'features': window_feat.values if window_feat is not None and not window_feat.empty else np.array([]),
                    'distributions': features['transaction_distributions'].get(window_idx, []),
                    'behavior_features': features['behavior_distributions'].get(window_idx, [])
                }
                window_data_list.append(window_data)
                
            # 计算动态风险值
            risk_values = self.risk_calculator.calculate_dynamic_risk_values(
                window_data_list,
                anomaly_features
            )
            
            # 存储结果
            risk_info = {
                'window_indices': window_indices,
                'risk_values': risk_values
            }
            
            risk_data[user_id] = risk_info
            
        self.processed_data['risk_data'] = risk_data
        return risk_data
        
    def update_detection_model(self):
        """更新检测模型参数"""
        if 'risk_data' not in self.processed_data:
            self.calculate_risk_values()
            
        risk_data = self.processed_data['risk_data']
        
        # 收集所有用户的风险值
        all_risk_values = []
        for user_id, risk_info in risk_data.items():
            all_risk_values.extend(risk_info['risk_values'])
            
        # 判断是否需要更新模型
        if self.model_updater.should_update(all_risk_values):
            # 更新模型参数
            updated_model = self.model_updater.update_model_parameters(all_risk_values)
            self.results['model_updated'] = True
            self.results['update_time'] = datetime.now()
            return updated_model
        else:
            self.results['model_updated'] = False
            return self.model_updater.get_current_model()
            
    def train_ensemble_classifier(self, X, y):
        """训练集成分类器
        
        参数:
            X: 特征数据
            y: 标签数据
        """
        self.ensemble_classifier.train(X, y)
        self.results['ensemble_trained'] = True
        self.results['ensemble_train_time'] = datetime.now()
        return self.ensemble_classifier
        
    def run_pipeline(self, X_train=None, y_train=None):
        """运行完整的欺诈检测流程
        
        参数:
            X_train: 训练特征数据（可选）
            y_train: 训练标签数据（可选）
            
        返回:
            流程运行结果
        """
        # 记录开始时间
        start_time = datetime.now()
        
        # 执行完整流程
        self.load_and_preprocess_data()
        self.segment_data_by_window()
        self.extract_window_features()
        self.analyze_behavior_consistency()
        self.calculate_anomaly_probabilities()
        self.calculate_risk_values()
        self.update_detection_model()
        
        # 如果提供了训练数据，训练集成分类器
        if X_train is not None and y_train is not None:
            self.train_ensemble_classifier(X_train, y_train)
        
        # 记录结束时间和总耗时
        end_time = datetime.now()
        self.results['start_time'] = start_time
        self.results['end_time'] = end_time
        self.results['total_time'] = (end_time - start_time).total_seconds()
        
        return self.results
        
    def detect_fraud(self, new_data):
        """对新数据进行欺诈检测
        
        参数:
            new_data: 新的交易和行为数据
            
        返回:
            欺诈检测结果
        """
        # 预处理新数据
        user_id = new_data.get('user_id')
        transaction_data = new_data.get('transactions', pd.DataFrame())
        behavior_data = new_data.get('behavior', pd.DataFrame())
        
        cleaned_trans = self.preprocessor.clean_transaction_data(transaction_data)
        cleaned_behav = self.preprocessor.clean_user_behavior_data(behavior_data)
        
        # 合并并提取特征
        combined_data = {
            'transactions': cleaned_trans,
            'behavior': cleaned_behav
        }
        
        window_data = self.window_processor.segment_user_data(combined_data)
        if not window_data:
            return {'user_id': user_id, 'fraud_probability': 0.0, 'is_fraud': False}
            
        # 获取最新窗口的数据
        latest_window_idx = max(window_data.keys())
        latest_window = window_data[latest_window_idx]
        
        # 提取特征
        features = self.feature_extractor.extract_window_features(latest_window)
        if features.empty:
            return {'user_id': user_id, 'fraud_probability': 0.0, 'is_fraud': False}
            
        # 使用集成分类器进行预测
        X = features.values.reshape(1, -1)
        fraud_prob = self.ensemble_classifier.predict_proba(X)[0]
        is_fraud = self.ensemble_classifier.predict(X)[0] == 1
        
        return {
            'user_id': user_id,
            'fraud_probability': float(fraud_prob),
            'is_fraud': bool(is_fraud),
            'detection_time': datetime.now()
        }
