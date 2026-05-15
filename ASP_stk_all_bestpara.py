import concurrent.futures
import datetime
import traceback
import random
import pandas as pd
from tqdm import tqdm
import rqdatac
from copy import deepcopy
from rqalpha_plus import run_func
from rqalpha_plus.apis import *
import talib

# 基础配置模板（benchmark 将被动态替换）
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
            "benchmark": "510300.XSHG",
            "record": True,
            "strategy_name": "ASP",
            "output_file": None,
            "plot": True,
            "plot_config": {
                "open_close_points": False,
                "weekly_indicators": False
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
    对单只股票进行参数寻优（5~35日），返回总收益率最优那次的结果。
    如果所有参数都失败，返回带错误信息的字典。
    """
    rqdatac.init()
    best_result = None
    best_period = None
    best_total_return = -float('inf')

    # 内部参数遍历，显示进度
    print(f"\n[{stock}] 开始参数寻优：周期 5 ~ 35")
    for period in range(5, 36):
        cfg = deepcopy(BASE_CONFIG)
        cfg["mod"]["sys_analyser"]["benchmark"] = stock
        cfg["mod"]["sys_analyser"]["strategy_name"] = f"{stock}---para:{period}"

        # 闭包捕获当前 period
        def init(context, p=period):
            context.s1 = stock
            context.LONGPERIOD = p

        def open_auction(context, bar_dict):
            try:
                turnover1 = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'total_turnover')
                long_avg1 = talib.LINEARREG_SLOPE(turnover1, timeperiod=context.LONGPERIOD)

                if long_avg1[-1] < 0 and long_avg1[-2] < 0:
                    pass
                if long_avg1[-1] < 0 and long_avg1[-2] > 0:
                    order_target_percent(context.s1, 0)
                if long_avg1[-1] >= 0 and long_avg1[-2] < 0:
                    order_target_percent(context.s1, 0.99)
                if long_avg1[-1] >= 0 and long_avg1[-2] >= 0:
                    pass
            except Exception:
                pass

        try:
            from rqalpha_plus import run_func
            result = run_func(config=cfg, init=init, open_auction=open_auction)
            summary = result.get('sys_analyser', {}).get('summary', {})
            total_ret = summary.get('total_returns', None)
            if total_ret is None:
                print(f"  参数 {period}: 回测成功但无收益数据，跳过")
                continue

            print(f"  参数 {period}: 总收益率 {total_ret:.4f}")
            if total_ret > best_total_return:
                best_total_return = total_ret
                best_period = period
                # 提取完整指标
                best_result = {
                    'order_book_id': stock,
                    'symbol': rqdatac.instruments(stock).symbol if stock else '',
                    'total_return': total_ret,
                    'annual_return': summary.get('annualized_returns', None),
                    'benchmark_total_return': summary.get('benchmark_total_returns', None),
                    'benchmark_annual_return': summary.get('benchmark_annualized_returns', None),
                    'excess_return': total_ret - (summary.get('benchmark_total_returns') or 0),
                    'sharpe': summary.get('sharpe', None),
                    'max_drawdown': summary.get('max_drawdown', None),
                    'volatility': summary.get('volatility', None),
                    'win_rate': summary.get('win_rate', None),
                    'total_trades': summary.get('total_trades', None),
                    'alpha': summary.get('alpha', None),
                    'beta': summary.get('beta', None),
                    'information_ratio': summary.get('information_ratio', None),
                    'tracking_error': summary.get('tracking_error', None),
                    'best_longperiod': best_period,
                }
        except Exception as e:
            print(f"  参数 {period}: 回测异常 {e}")

    if best_result is not None:
        print(f"[{stock}] 最优参数 {best_period}，总收益率 {best_total_return:.4f}")
        return best_result
    else:
        # 全部参数失败
        return {
            'order_book_id': stock,
            'error': f"全部参数（5-35）回测失败或收益缺失",
            'traceback': ""
        }


def main():
    # 1. 随机选取10只股票测试
    all_stocks = get_stock_universe('2026-03-01')
    stocks = random.sample(all_stocks, min(10, len(all_stocks)))
    print(f"随机选取 {len(stocks)} 只股票进行回测：{stocks}")

    # 2. 多进程执行
    results = []
    failed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # 提交所有任务
        futures = {executor.submit(run_bt, stock): stock for stock in stocks}
        # 外部进度条（按股票数量）
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(stocks), desc="总体进度"):
            stock = futures[fut]
            try:
                res = fut.result()
                if 'error' in res:
                    failed.append(res)
                else:
                    results.append(res)
            except Exception as exc:
                failed.append({'order_book_id': stock, 'error': str(exc), 'traceback': traceback.format_exc()})

    # 3. 保存结果
    if results:
        df = pd.DataFrame(results)
        output_path = f"asp_batch_backtest_{datetime.date.today().strftime('%Y%m%d')}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"成功回测 {len(results)} 只，结果已保存至 {output_path}")
    else:
        print("没有成功的结果可保存。")

    if failed:
        error_df = pd.DataFrame(failed)
        err_path = f"asp_errors_{datetime.date.today().strftime('%Y%m%d')}.csv"
        error_df.to_csv(err_path, index=False, encoding='utf-8-sig')
        print(f"回测失败 {len(failed)} 只，详见 {err_path}")
    else:
        print("全部回测成功！")


if __name__ == "__main__":
    main()