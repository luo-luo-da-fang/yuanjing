import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="真实行情预测看板",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 全球市场真实行情与趋势预测看板")
st.caption("数据来源：雅虎财经 | 仅供学习研究，不构成投资建议")
st.divider()

# ==================== 辅助函数：股票代码格式补全 ====================
def format_stock_code(code):
    """自动识别市场，补全股票代码后缀"""
    code = code.strip()
    # 已带后缀直接返回
    if "." in code:
        return code
    # 纯数字判断市场
    if code.isdigit():
        if len(code) == 6:
            # A股：6/688开头为沪市，0/3开头为深市
            if code.startswith(("6", "688")):
                return f"{code}.SS"
            else:
                return f"{code}.SZ"
        elif len(code) <= 5:
            # 港股自动补零到5位
            return f"{code.zfill(5)}.HK"
    # 纯字母默认按美股处理
    return code

# ==================== 缓存数据获取函数 ====================
@st.cache_data(ttl=3600)  # 缓存1小时，避免频繁请求
def get_stock_history(code, period="180d", interval="1d"):
    """获取股票历史行情数据"""
    ticker = yf.Ticker(code)
    df = ticker.history(period=period, interval=interval, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    # 重置索引、去除时区、统一中文列名
    df = df.reset_index()
    df["日期"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.rename(columns={
        "Open": "开盘",
        "High": "最高",
        "Low": "最低",
        "Close": "收盘",
        "Volume": "成交量"
    })
    df = df.sort_values("日期").reset_index(drop=True)
    return df

# ==================== 侧边栏交互控件 ====================
with st.sidebar:
    st.subheader("选股与参数设置")
    
    # 热门股票快速选择
    hot_stocks = {
        "000001 平安银行": "000001",
        "000002 万科A": "000002",
        "600519 贵州茅台": "600519",
        "300750 宁德时代": "300750",
        "00700 腾讯控股": "00700",
        "AAPL 苹果": "AAPL",
        "TSLA 特斯拉": "TSLA"
    }
    selected_hot = st.selectbox("热门股票快速选择", list(hot_stocks.keys()), index=1)
    
    # 支持手动输入任意代码
    manual_code = st.text_input("手动输入股票代码", value=hot_stocks[selected_hot],
                               help="A股输6位代码、港股输5位代码、美股输英文代码")
    stock_code = format_stock_code(manual_code)

    # 历史行情周期选择
    period_options = {
        "近1个月": "30d",
        "近3个月": "90d",
        "近半年": "180d",
        "近1年": "1y",
        "近3年": "3y"
    }
    selected_period = st.selectbox("历史行情周期", list(period_options.keys()), index=2)
    period_value = period_options[selected_period]

    # 预测天数调节
    predict_days = st.slider("趋势预测天数", min_value=1, max_value=30, value=7)

    st.divider()
    st.caption("数据每1小时自动更新一次")

# ==================== 主内容区 ====================
try:
    # 获取历史数据
    history_df = get_stock_history(stock_code, period_value)
    
    if history_df.empty:
        st.warning("未查询到该股票数据，请检查股票代码后重试")
    else:
        # ========== 1. 核心指标卡片 ==========
        latest = history_df.iloc[-1]
        prev = history_df.iloc[-2]
        change_pct = round((latest["收盘"] - prev["收盘"]) / prev["收盘"] * 100, 2)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盘价", f"{round(latest['收盘'], 2)} 元", f"{change_pct}%")
        col2.metric("当日最高", f"{round(latest['最高'], 2)} 元")
        col3.metric("当日最低", f"{round(latest['最低'], 2)} 元")
        col4.metric("当日成交量", f"{int(latest['成交量']):,}")
        
        st.divider()

        # ========== 2. 多标签页功能 ==========
        tab1, tab2, tab3, tab4 = st.tabs(["K线行情", "历史数据", "技术指标", "趋势预测"])
        
        with tab1:
            st.subheader(f"{manual_code} K线走势")
            fig = go.Figure(data=[go.Candlestick(
                x=history_df["日期"],
                open=history_df["开盘"],
                high=history_df["最高"],
                low=history_df["最低"],
                close=history_df["收盘"],
                name="K线"
            )])
            fig.update_layout(xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("历史行情明细")
            display_df = history_df[["日期", "开盘", "最高", "最低", "收盘", "成交量"]].copy()
            display_df = display_df.sort_values("日期", ascending=False).reset_index(drop=True)
            numeric_cols = ["开盘", "最高", "最低", "收盘"]
            display_df[numeric_cols] = display_df[numeric_cols].round(2)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        with tab3:
            st.subheader("均线技术指标")
            df_indicator = history_df.copy()
            df_indicator["MA5"] = df_indicator["收盘"].rolling(5).mean()
            df_indicator["MA10"] = df_indicator["收盘"].rolling(10).mean()
            df_indicator["MA20"] = df_indicator["收盘"].rolling(20).mean()
            df_indicator["MA60"] = df_indicator["收盘"].rolling(60).mean()
            
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["收盘"], name="收盘价", line_color="#1f77b4"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA5"], name="MA5", line_color="#ff7f0e"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA10"], name="MA10", line_color="#2ca02c"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA20"], name="MA20", line_color="#d62728"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA60"], name="MA60", line_color="#9467bd"))
            fig_ma.update_layout(height=500, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_ma, use_container_width=True)
            
            st.caption("MA5/10/20/60 分别代表5日、10日、20日、60日移动平均线")

        with tab4:
            st.subheader(f"未来{predict_days}天趋势预测（线性回归模型）")
            st.info("⚠️ 本预测仅基于历史价格的数学趋势拟合，不代表真实未来走势，股市有风险，投资需谨慎")
            
            # 数据准备：用日期序号作为特征
            close_prices = history_df["收盘"].values.reshape(-1, 1)
            days_index = np.arange(len(close_prices)).reshape(-1, 1)
            
            # 训练线性回归模型
            model = LinearRegression()
            model.fit(days_index, close_prices)
            
            # 生成未来预测序列
            future_index = np.arange(len(close_prices), len(close_prices)+predict_days).reshape(-1, 1)
            future_pred = model.predict(future_index).flatten()
            
            # 按交易日生成预测日期
            last_date = history_df["日期"].iloc[-1]
            future_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=predict_days)
            
            # 绘制预测对比图
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=history_df["日期"], y=history_df["收盘"],
                name="历史收盘价", line_color="#1f77b4"
            ))
            fig_pred.add_trace(go.Scatter(
                x=future_dates, y=future_pred,
                name="预测趋势", line_color="#ff4b5c", line_dash="dash"
            ))
            fig_pred.update_layout(height=500)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 预测结果表格
            pred_df = pd.DataFrame({
                "预测日期": future_dates.strftime("%Y-%m-%d"),
                "预测收盘价": np.round(future_pred, 2)
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"数据加载失败：{str(e)}")
    st.caption("请检查股票代码是否正确，或稍后重试")
