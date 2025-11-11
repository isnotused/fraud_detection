
# 因为 streamlit 默认开启多页导航栏模式，有两种方法关闭它：

1. 在启动时 streamlit run head.py --client.showSidebarNavigation false

2. 在 config.toml 配置文件中修改：showSidebarNavigation = false；终端命令下输入 streamlit config show