"""
股票预测大数据应用 - Flask Web 应用
提供股票数据展示、分析和预测功能
"""

from flask import Flask, render_template, jsonify, request
import random
import datetime
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from stock_data_generator import StockDataGenerator
from stock_predictor import StockPredictor

app = Flask(__name__)

# 初始化股票数据生成器和预测器
stock_gen = StockDataGenerator()
predictor = StockPredictor()

@app.route('/')
def index():
    """主页"""
    return render_template('stock_index.html')

@app.route('/api/stocks')
def get_stocks():
    """获取所有股票列表"""
    stocks = stock_gen.get_stock_list()
    return jsonify(stocks)

@app.route('/api/stock/<stock_code>')
def get_stock_data(stock_code):
    """获取指定股票数据"""
    data = stock_gen.get_stock_data(stock_code)
    return jsonify(data)

@app.route('/api/stock/<stock_code>/history')
def get_stock_history(stock_code):
    """获取股票历史数据"""
    days = request.args.get('days', 30, type=int)
    data = stock_gen.get_stock_history(stock_code, days)
    return jsonify(data)

@app.route('/api/stock/<stock_code>/predict')
def predict_stock(stock_code):
    """预测股票涨幅"""
    days = request.args.get('days', 7, type=int)
    prediction = predictor.predict(stock_code, days)
    return jsonify(prediction)

@app.route('/api/stock/<stock_code>/analysis')
def analyze_stock(stock_code):
    """分析股票"""
    analysis = predictor.analyze(stock_code)
    return jsonify(analysis)

@app.route('/api/stock/<stock_code>/indicators')
def get_indicators(stock_code):
    """获取技术指标"""
    indicators = stock_gen.get_technical_indicators(stock_code)
    return jsonify(indicators)

@app.route('/api/stock/refresh', methods=['POST'])
def refresh_stock_data():
    """刷新股票数据"""
    stock_gen.generate_new_data()
    return jsonify({'status': 'success', 'message': '股票数据已更新'})

if __name__ == '__main__':
    print("=" * 60)
    print("股票预测大数据应用启动中...")
    print("=" * 60)
    
    # 生成初始股票数据
    print("正在生成股票数据...")
    stock_gen.generate_initial_data()
    stocks = stock_gen.get_stock_list()
    print(f"✓ 已生成 {len(stocks)} 只股票数据")
    
    print("\n正在初始化预测模型...")
    predictor.initialize(stock_gen)
    print("✓ 预测模型初始化完成")
    
    print("\n" + "=" * 60)
    print("股票预测应用已就绪!")
    print("访问地址：http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
