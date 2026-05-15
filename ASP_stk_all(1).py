import concurrent.futures
import datetime
import traceback
import pandas as pd
from tqdm import tqdm
import rqdatac
from copy import deepcopy
import random
from rqalpha_plus import run_func
from rqalpha_plus.apis import *
import talib

# 基础回测配置模板（将动态修改 benchmark）
BASE_CONFIG = {
    "base": {
        "start_date": "2025-04-01",
        "end_date": "2026-04-01",
        "frequency": "1d",
        "accounts": {
            "stock": 1000000
        },
        "log_level": "INFO",
        "data_bundle_path": None,
    },
    "mod": {
        "sys_simulation": {
            "enabled": True,
            "volume_limit": False,
            "volume_percent": 0.3,
        },
        "sys_transaction_cost": {
            "cn_stock_min_commission": 0,
            "stock_commission_multiplier": 0.25,
            "futures_commission_multiplier": 1,
            "tax_multiplier": 1,
            "pit_tax": False,
        },
        "sys_analyser": {
            "enabled": True,
            "benchmark": "510300.XSHG",  # 将在每只股票中替换为自身
            "record": True,
            "strategy_name": "ASP",
            "output_file": None,
            "plot": True,
            "plot_config": {
                "open_close_points": False,
                "weekly_indicators": True
            },
        }
    },
    "extra": {
        "log_level": "error",  # 多进程时抑制详细日志
    }
}


def get_stock_universe(as_of_date='2026-03-01'):
    """获取沪深A股中上市满一年且Active的全集"""
    rqdatac.init()
    end_date = datetime.datetime.strptime(as_of_date, '%Y-%m-%d')
    one_year_ago = end_date - datetime.timedelta(days=365)

    all_stocks = rqdatac.all_instruments(type='CS', market='cn')
    # 清洗上市日期范围
    all_stocks = all_stocks[
        (all_stocks['listed_date'].astype(str) > '19900101') &
        (all_stocks['listed_date'].astype(str) < '21000101')
        ].copy()
    all_stocks['listed_date'] = pd.to_datetime(all_stocks['listed_date'])

    universe = all_stocks[
        (all_stocks['listed_date'] < one_year_ago) &
        (all_stocks['status'] == 'Active')
        ]
    return universe['order_book_id'].tolist()


def run_bt(stock):
    """
    对单只股票执行回测，返回结果字典；若失败返回 None 及错误信息。
    """
    # 每个子进程独立初始化 RQData
    # rqdatac.init()

    # 深拷贝配置，避免模板污染
    cfg = deepcopy(BASE_CONFIG)
    cfg["mod"]["sys_analyser"]["benchmark"] = stock
    cfg["mod"]["sys_analyser"]["strategy_name"] = stock

    # 策略的 init 和 open_auction 使用闭包捕获 stock 代码
    def init(context):
        context.s1 = stock
        context.LONGPERIOD = 15

    def open_auction(context, bar_dict):
        # 使用原策略：15日成交额线性回归斜率
        try:
            turnover1 = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'total_turnover')
            long_avg1 = talib.LINEARREG_SLOPE(turnover1, timeperiod=15)

            if long_avg1[-1] < 0 and long_avg1[-2] < 0:
                pass
            if long_avg1[-1] < 0 and long_avg1[-2] > 0:
                order_target_percent(context.s1, 0)
            if long_avg1[-1] >= 0 and long_avg1[-2] < 0:
                order_target_percent(context.s1, 0.99)
            if long_avg1[-1] >= 0 and long_avg1[-2] >= 0:
                pass
        except Exception:
            # 数据不足时静默跳过，避免打断回测
            pass

    try:
        # 注意：此处需确保多进程环境中正确导入 run_func
        from rqalpha_plus import run_func
        result = run_func(config=cfg, init=init, open_auction=open_auction)
        summary = result.get('sys_analyser', {}).get('summary', {})

        # 提取所需指标
        total_return = summary.get('total_returns', 0)
        annual_return = summary.get('annualized_returns', 0)
        benchmark_total_return = summary.get('benchmark_total_returns', 0)
        benchmark_annual_return = summary.get('benchmark_annualized_returns', 0)
        excess_return = total_return - benchmark_total_return

        return {
            'order_book_id': stock,
            'symbol': rqdatac.instruments(stock).symbol if stock else '',
            'total_return': total_return,
            'annual_return': annual_return,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return,
            'excess_return': excess_return,
            'sharpe': summary.get('sharpe', None),
            'max_drawdown': summary.get('max_drawdown', None),
            'volatility': summary.get('volatility', None),
            'win_rate': summary.get('win_rate', None),
            'total_trades': summary.get('total_trades', None),
            'alpha': summary.get('alpha', None),
            'beta': summary.get('beta', None),
            'information_ratio': summary.get('information_ratio', None),
            'tracking_error': summary.get('tracking_error', None),
        }
    except Exception as e:
        return {
            'order_book_id': stock,
            'error': f"{type(e).__name__}: {e}",
            'traceback': traceback.format_exc()
        }


def main():
    # 1. 获取股票池
    # 1. 获取全部符合条件的股票，并随机选 10 只测试
    all_stocks = get_stock_universe('2026-03-01')
    # print(f"将回测 {len(stocks)} 只上市满一年的 A 股")
    stocks = random.sample(all_stocks, min(10, len(all_stocks)))
    print(f"随机选取 {len(stocks)} 只股票进行回测：{stocks}")

    # 2. 多进程执行
    results = []
    failed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        # 使用 tqdm 显示进度
        futures = {executor.submit(run_bt, stock): stock for stock in stocks}
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(stocks)):
            stock = futures[fut]
            try:
                res = fut.result()
                if 'error' in res:
                    failed.append(res)
                else:
                    results.append(res)
            except Exception as exc:
                failed.append({'order_book_id': stock, 'error': str(exc)})

    # 3. 保存结果
    if results:
        df = pd.DataFrame(results)
        output_path = f"asp_batch_backtest_{datetime.date.today().strftime('%Y%m%d')}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"成功回测 {len(results)} 只，结果已保存至 {output_path}")
    else:
        print("没有成功的结果可保存。")

    # 4. 错误日志
    if failed:
        error_df = pd.DataFrame(failed)
        err_path = f"asp_errors_{datetime.date.today().strftime('%Y%m%d')}.csv"
        error_df.to_csv(err_path, index=False, encoding='utf-8-sig')
        print(f"回测失败 {len(failed)} 只，详见 {err_path}")
    else:
        print("全部回测成功！")


if __name__ == "__main__":
    main()