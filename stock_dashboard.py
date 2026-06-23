import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="真实A股行情预测看板",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 A股真实行情与趋势预测看板")
st.caption("数据来源：东方财富 | 仅供学习研究，不构成投资建议")
st.divider()

# ==================== 缓存数据获取函数 ====================
@st.cache_data(ttl=3600)  # 缓存1小时，避免频繁请求接口
def get_all_stock_list():
    """获取全A股股票代码+名称列表"""
    df = ak.stock_info_a_code_name()
    df["label"] = df["code"] + " " + df["name"]
    return df

@st.cache_data(ttl=1800)
def get_stock_history(code, start_date, end_date, adjust="qfq"):
    """获取单只股票历史日线数据"""
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust=adjust
    )
    # 格式化日期列，方便后续处理
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.sort_values("日期").reset_index(drop=True)
    return df

# ==================== 侧边栏交互控件 ====================
with st.sidebar:
    st.subheader("选股与参数设置")
    
    # 1. 股票选择
    try:
        stock_df = get_all_stock_list()
        stock_options = stock_df["label"].tolist()
        # 默认选中平安银行作为示例
        default_idx = stock_options.index("000001 平安银行") if "000001 平安银行" in stock_options else 0
        selected = st.selectbox("选择股票", stock_options, index=default_idx)
        stock_code = selected.split(" ")[0]
        stock_name = selected.split(" ")[1]
    except Exception as e:
        st.error("股票列表加载失败，请刷新重试")
        stock_code = "000001"
        stock_name = "平安银行"

    # 2. 时间范围
    end_default = datetime.now().strftime("%Y%m%d")
    start_default = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.text_input("开始日期", value=start_default, help="格式：YYYYMMDD")
    with col2:
        end_date = st.text_input("结束日期", value=end_default, help="格式：YYYYMMDD")

    # 3. 复权方式
    adjust_type = st.selectbox(
        "复权方式",
        ["qfq", "hfq", ""],
        format_func=lambda x: {"qfq":"前复权","hfq":"后复权","":"不复权"}[x],
        index=0
    )

    # 4. 预测天数
    predict_days = st.slider("趋势预测天数", min_value=1, max_value=30, value=7)

    st.divider()
    st.caption("数据每30分钟自动更新一次")

# ==================== 主内容区 ====================
try:
    # 获取历史数据
    history_df = get_stock_history(stock_code, start_date, end_date, adjust_type)
    
    if history_df.empty:
        st.warning("未查询到该时间段数据，请调整日期后重试")
    else:
        # ========== 1. 核心指标卡片 ==========
        latest = history_df.iloc[-1]
        prev = history_df.iloc[-2]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("最新收盘价", f"{latest['收盘']} 元", f"{latest['涨跌幅']}%")
        col2.metric("当日最高", f"{latest['最高']} 元")
        col3.metric("当日最低", f"{latest['最低']} 元")
        col4.metric("换手率", f"{latest['换手率']}%")
        
        st.divider()

        # ========== 2. K线走势图 ==========
        tab1, tab2, tab3, tab4 = st.tabs(["K线行情", "历史数据", "技术指标", "趋势预测"])
        
        with tab1:
            st.subheader(f"{stock_name}（{stock_code}）K线走势")
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

        # ========== 3. 历史数据明细 ==========
        with tab2:
            st.subheader("历史行情明细")
            # 按日期倒序展示
            st.dataframe(
                history_df.sort_values("日期", ascending=False),
                use_container_width=True,
                hide_index=True
            )

        # ========== 4. 技术指标 ==========
        with tab3:
            st.subheader("常用技术指标")
            # 计算移动平均线
            df_indicator = history_df.copy()
            df_indicator["MA5"] = df_indicator["收盘"].rolling(5).mean()
            df_indicator["MA10"] = df_indicator["收盘"].rolling(10).mean()
            df_indicator["MA20"] = df_indicator["收盘"].rolling(20).mean()
            df_indicator["MA60"] = df_indicator["收盘"].rolling(60).mean()
            
            # 绘制均线图
            fig_ma = go.Figure()
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["收盘"], name="收盘价", line_color="#1f77b4"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA5"], name="MA5", line_color="#ff7f0e"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA10"], name="MA10", line_color="#2ca02c"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA20"], name="MA20", line_color="#d62728"))
            fig_ma.add_trace(go.Scatter(x=df_indicator["日期"], y=df_indicator["MA60"], name="MA60", line_color="#9467bd"))
            fig_ma.update_layout(height=500, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_ma, use_container_width=True)
            
            st.caption("MA5/10/20/60 分别代表5日、10日、20日、60日移动平均线")

        # ========== 5. 线性回归趋势预测 ==========
        with tab4:
            st.subheader(f"未来{predict_days}天趋势预测（线性回归模型）")
            st.info("⚠️ 本预测仅基于历史价格的数学趋势拟合，不代表真实未来走势，股市有风险，投资需谨慎")
            
            # 数据准备：用日期序号作为特征，收盘价作为标签
            close_prices = history_df["收盘"].values.reshape(-1, 1)
            days_index = np.arange(len(close_prices)).reshape(-1, 1)
            
            # 训练线性回归模型
            model = LinearRegression()
            model.fit(days_index, close_prices)
            
            # 生成未来天数的序号
            future_index = np.arange(len(close_prices), len(close_prices)+predict_days).reshape(-1, 1)
            future_pred = model.predict(future_index).flatten()
            
            # 构造预测结果日期
            last_date = history_df["日期"].iloc[-1]
            future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=predict_days, freq="B")
            
            # 绘制预测图
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
                "预测收盘价(元)": np.round(future_pred, 2)
            })
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"数据加载失败：{str(e)}")
    st.caption("可能原因：网络波动、日期格式错误或接口限流，请稍后重试")
