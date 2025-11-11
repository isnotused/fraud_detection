import pages as pgs

import streamlit as st
from streamlit_option_menu import option_menu  
st.set_page_config(page_title='test', page_icon=' ', layout='wide')

with st.sidebar:
    page = option_menu(
        menu_title='基于大数据的金融领域消费欺诈检测系统',
        options=['首页', '用户数据', '风险分析评估', '系统管理'],
        default_index=0,
        menu_icon='windows',
        icons=['house', 'people', 'piggy-bank', 'gear']    # 图标参考 https://icons.getbootstrap.com/ 
    )

functions = {
    '首页': pgs.home,
    '用户数据': pgs.users_data_app,
    '风险分析评估': pgs.data_analyzer_app,
    '系统管理': pgs.system_management
}

go_to = functions.get(page)

if go_to:
    go_to()
