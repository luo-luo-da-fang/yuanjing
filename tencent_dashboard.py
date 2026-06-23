"""
股票预测大数据应用 - Streamlit 版本
提供股票数据展示、分析和预测功能
"""
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from stock_data_generator import StockDataGenerator
from stock_predictor import StockPredictor

# 页面基础配置
st.set_page_config(page_title="股票预测大数据平台", layout="wide")

# 缓存初始化模块，避免每次交互重复加载
@st.cache_resource
def init_service():
    stock_gen = StockDataGenerator()
    predictor = StockPredictor()
    # 生成初始股票数据
    stock_gen.generate_initial_data()
    predictor.initialize(stock_gen)
    return stock_gen, predictor

stock_gen, predictor = init_service()

# 页面标题
st.title("📈 股票预测大数据应用")
st.divider()

# 侧边栏操作面板
with st.sidebar:
    st.subheader("操作面板")
    # 股票选择
    stock_list = stock_gen.get_stock_list()
    selected_stock = st.selectbox("选择股票代码", stock_list)
    
    # 参数配置
    predict_days = st.slider("预测天数", min_value=1, max_value=30, value=7)
    history_days = st.slider("历史数据周期", min_value=7, max_value=180, value=30)
    
    # 刷新按钮
    if st.button("🔄 刷新股票数据", use_container_width=True):
        stock_gen.generate_new_data()
        st.cache_resource.clear()
        st.success("股票数据已更新")

# 主内容区 - 多标签页展示
tab1, tab2, tab3, tab4 = st.tabs(["历史行情", "走势预测", "深度分析", "技术指标"])

with tab1:
    st.subheader(f"{selected_stock} 历史行情")
    history_data = stock_gen.get_stock_history(selected_stock, history_days)
    st.dataframe(history_data, use_container_width=True)

with tab2:
    st.subheader(f"{selected_stock} 未来{predict_days}天走势预测")
    prediction = predictor.predict(selected_stock, predict_days)
    st.json(prediction)

with tab3:
    st.subheader(f"{selected_stock} 深度分析报告")
    analysis = predictor.analyze(selected_stock)
    st.json(analysis)

with tab4:
    st.subheader(f"{selected_stock} 技术指标")
    indicators = stock_gen.get_technical_indicators(selected_stock)
    st.json(indicators)
