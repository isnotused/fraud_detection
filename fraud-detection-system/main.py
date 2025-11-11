import argparse
import json
import logging
from datetime import datetime
import pandas as pd

from pipeline.detection_pipeline import FraudDetectionPipeline

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('fraud_detection_system')

def load_config(config_path):
    """加载配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        配置字典
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {str(e)}")
        raise

def load_training_data(train_data_path):
    """加载训练数据
    
    参数:
        train_data_path: 训练数据路径
        
    返回:
        特征数据X和标签数据y
    """
    try:
        if train_data_path.endswith('.csv'):
            data = pd.read_csv(train_data_path)
        elif train_data_path.endswith('.parquet'):
            data = pd.read_parquet(train_data_path)
        else:
            raise ValueError("不支持的训练数据格式")
            
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        logger.info(f"成功加载训练数据，共 {X.shape[0]} 个样本，{X.shape[1]} 个特征")
        return X, y
    except Exception as e:
        logger.error(f"加载训练数据失败: {str(e)}")
        raise

def run_batch_processing(config):
    """运行批量处理模式
    
    参数:
        config: 配置字典
    """
    logger.info("开始批量欺诈检测处理")
    
    # 初始化检测管道
    pipeline = FraudDetectionPipeline(config)
    
    # 加载训练数据
    X_train, y_train = None, None
    if 'train_data_path' in config and config['train_data_path']:
        X_train, y_train = load_training_data(config['train_data_path'])
    
    # 运行完整流程
    results = pipeline.run_pipeline(X_train, y_train)
    
    # 记录结果
    logger.info(f"批量处理完成，总耗时: {results['total_time']:.2f} 秒")
    logger.info(f"模型是否更新: {'是' if results.get('model_updated', False) else '否'}")
    
    # 保存结果
    if 'output_path' in config and config['output_path']:
        try:
            with open(config['output_path'], 'w') as f:
                json.dump(results, f, default=str)
            logger.info(f"处理结果已保存至: {config['output_path']}")
        except Exception as e:
            logger.error(f"保存结果失败: {str(e)}")

def run_realtime_detection(config, input_data):
    """运行实时检测模式
    
    参数:
        config: 配置字典
        input_data: 输入的实时数据
        
    返回:
        检测结果
    """
    logger.info("开始实时欺诈检测")
    
    # 初始化检测管道
    pipeline = FraudDetectionPipeline(config)
    
    # 确保管道已初始化
    pipeline.load_and_preprocess_data()
    pipeline.segment_data_by_window()
    pipeline.extract_window_features()
    
    # 如果需要，更新模型
    pipeline.calculate_risk_values()
    pipeline.update_detection_model()
    
    # 加载训练数据并训练集成分类器
    if 'train_data_path' in config and config['train_data_path']:
        X_train, y_train = load_training_data(config['train_data_path'])
        pipeline.train_ensemble_classifier(X_train, y_train)
    
    # 执行检测
    result = pipeline.detect_fraud(input_data)
    
    logger.info(f"实时检测完成: 用户 {result['user_id']}, 欺诈概率: {result['fraud_probability']:.4f}")
    return result

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='金融消费欺诈检测系统')
    parser.add_argument('--config', required=True, help='配置文件路径')
    parser.add_argument('--mode', choices=['batch', 'realtime'], default='batch', 
                      help='运行模式: batch（批量处理）或 realtime（实时检测）')
    parser.add_argument('--input', help='实时检测的输入数据路径（仅在realtime模式下使用）')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    if args.mode == 'batch':
        # 运行批量处理
        run_batch_processing(config)
    else:
        # 运行实时检测
        if not args.input:
            logger.error("实时检测模式需要指定输入数据路径")
            return
            
        # 加载输入数据
        try:
            with open(args.input, 'r') as f:
                input_data = json.load(f)
            logger.info(f"成功加载实时检测数据: {args.input}")
        except Exception as e:
            logger.error(f"加载实时检测数据失败: {str(e)}")
            return
            
        # 执行实时检测
        result = run_realtime_detection(config, input_data)
        
        # 输出结果
        print(json.dumps(result, default=str, indent=2))
        
        # 保存结果
        if 'realtime_output_path' in config and config['realtime_output_path']:
            try:
                with open(config['realtime_output_path'], 'w') as f:
                    json.dump(result, f, default=str, indent=2)
                logger.info(f"实时检测结果已保存至: {config['realtime_output_path']}")
            except Exception as e:
                logger.error(f"保存实时检测结果失败: {str(e)}")

if __name__ == "__main__":
    main()
