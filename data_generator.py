import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from progress_bar import update_progress

def generate_financial_transaction_data(num_users=1000, num_days=30, transactions_per_day=10000):
    """金融消费交易数据"""
    progress = 0
    update_progress(progress, "交易数据")
    
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
            progress = count / total
            update_progress(progress, "交易数据")
            if count % 1000 == 0:
                import time
                time.sleep(0.01)
    
    df = pd.DataFrame(data)
    df = df.sort_values(["user_id", "transaction_time"])
    update_progress(1, "交易数据")
    return df

def generate_user_behavior_data(transaction_data):
    """多维用户行为数据"""
    progress = 0
    update_progress(progress, "用户行为数据")
    
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
        
        progress = (i + 1) / total_users
        update_progress(progress, "用户行为数据")
        if i % 100 == 0:
            import time
            time.sleep(0.01)
    
    df = pd.DataFrame(behaviors)
    update_progress(1, "用户行为数据")
    return df

def save_data(transaction_df, behavior_df, trans_path="transactions.csv", behavior_path="user_behaviors.csv"):
    """保存数据到CSV文件"""
    transaction_df.to_csv(trans_path, index=False)
    behavior_df.to_csv(behavior_path, index=False)
    print(f"交易数据已保存到 {trans_path}")
    print(f"用户行为数据已保存到 {behavior_path}")
    return trans_path, behavior_path

def load_data(trans_path="transactions.csv", behavior_path="user_behaviors.csv"):
    """从CSV文件加载数据"""
    print(f"从 {trans_path} 加载交易数据")
    trans_df = pd.read_csv(trans_path, parse_dates=["transaction_time"])
    
    print(f"从 {behavior_path} 加载用户行为数据")
    behavior_df = pd.read_csv(behavior_path)
    
    return trans_df, behavior_df