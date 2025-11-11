import sys
import time

def update_progress(progress, task_name="处理中"):
    """更新进度条"""
    bar_length = 50
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "进度错误\r\n"
    if progress < 0:
        progress = 0
        status = "准备中...\r\n"
    if progress >= 1:
        progress = 1
        status = "完成!\r\n"
    block = int(round(bar_length * progress))
    text = f"\r{task_name}: [{'#' * block + '-' * (bar_length - block)}] {progress * 100:.2f}% {status}"
    sys.stdout.write(text)
    sys.stdout.flush()
    return progress