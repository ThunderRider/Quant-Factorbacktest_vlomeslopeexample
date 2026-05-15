from rqalpha_plus.apis import *
import talib
import pandas as pd
import rqdatac
from datetime import datetime, timedelta
from copy import deepcopy

rqdatac.init()

BASE_CONFIG = {
    "base": {
        "start_date": "2025-04-01",  # 回测开始日期
        "end_date": "2026-04-01",  # 回测结束日期
        "frequency": "1d",
        "accounts": {
            "stock": 100000
        },  # 设置初始资金，此处类别需要填写为 stock
        "log_level": "INFO",
        "data_bundle_path": None,
    },
    "mod": {
        "sys_simulation": {
            "enabled": True,
            "volume_limit": False,  # 是否开启成交量限制
            'volume_percent': 0.3,  # 按照 bar 数据成交量的一定比例进行限制，超限部分无法在当前 bar 一次性撮合成交
        },
        "sys_transaction_cost": {
            "cn_stock_min_commission": 0, # 股票最小手续费，单位元
            # "commission_multiplier": 0.25, # 佣金倍率（即将废弃）
            "stock_commission_multiplier": 0.25, # 股票佣金倍率,即在默认的手续费率基础上按该倍数进行调整，股票的默认佣金为万八
            "futures_commission_multiplier": 1, # 期货佣金倍率,即在默认的手续费率基础上按该倍数进行调整，期货默认佣金因合约而异
            "tax_multiplier": 1, # 印花倍率，即在默认的印花税基础上按该倍数进行调整，股票默认印花税为万分之五，单边收取
            "pit_tax": False, # 是否使用回测当时时间点对应的真实印花税率
        },
        "sys_analyser": {
            "enabled": True,
            "benchmark": "510300.XSHG",  # 策略基准合约
            "record": True, # 当不输出 csv/pickle/plot 等内容时，关闭该项可关闭策略运行过程中部分收集数据的逻辑，用以提升性能
            "strategy_name": "ASP", # 策略名称，可设置 summary 报告中的 strategy_name 字段，并展示在 plot 回测结果图中
            "output_file": None, # 回测结果输出的文件路径，该文件为 pickle 格式，内容为每日净值、头寸、流水及风险指标等；若不设置则不输出该文件
            # "report_save_path": rf"C:\Users\39734\OneDrive\Documents\5.工作需要\3.HF.工作所需\9.RiceQuant\ASP_reports", # 回测报告的数据目录，报告为 csv 格式；若不设置则不输出报告
            "plot": True,  # 是否画出回测结果收益图
            'plot_config': {
                # 是否在收益图中展示买卖点
                'open_close_points': True,
                # 是否在收益图中展示周度指标和收益曲线
                'weekly_indicators': False
            },
        }
    },
}

stock = "600679.XSHG"
period = 13
cfg = deepcopy(BASE_CONFIG)
cfg["mod"]["sys_analyser"]["benchmark"] = stock
cfg["mod"]["sys_analyser"]["strategy_name"] = f"{stock}---para:{period}"


def init(context):
    context.s1 = stock
    context.LONGPERIOD = period


def open_auction(context, bar_dict):
    # 1
    turnover1 = history_bars(context.s1, context.LONGPERIOD + 1, '1d', 'total_turnover')
    long_avg1 = talib.LINEARREG_SLOPE(turnover1, timeperiod=context.LONGPERIOD)
    if long_avg1[-1] < 0 and  long_avg1[-2] < 0:
        pass
    if long_avg1[-1] < 0 and  long_avg1[-2] > 0:
        order_target_percent(context.s1, 0)
    if long_avg1[-1] >= 0 and  long_avg1[-2] < 0:
        order_target_percent(context.s1, 0.99)
    if long_avg1[-1] >= 0 and  long_avg1[-2] >= 0:
        pass


if __name__ == "__main__":
    from rqalpha_plus import run_func
    run_func(config=cfg, init=init, open_auction=open_auction)
    # result = run_func(config=config, init=init, open_auction=open_auction)
    # print(result)
    # print(result['sys_analyser']['summary']['sharpe'])

