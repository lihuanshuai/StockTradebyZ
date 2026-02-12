from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from xtquant import xtdata

from fetch_kline import filter_by_boards_stocklist, load_stocklist
from Selector import Selector

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------


def load_data(codes: list[str]) -> dict[str, pd.DataFrame]:
    fields = ["open", "high", "low", "close", "volume"]
    data: dict[str, pd.DataFrame] = xtdata.get_market_data(
        field_list=fields,
        stock_list=codes,
        period="1d",
        dividend_type="front",
    )
    result = {}
    for code in codes:
        df_open = data["open"].T[code]
        df_high = data["high"].T[code]
        df_low = data["low"].T[code]
        df_close = data["close"].T[code]
        df_volume = data["volume"].T[code]
        df = pd.DataFrame(
            {
                "open": df_open,
                "high": df_high,
                "low": df_low,
                "close": df_close,
                "volume": df_volume,
            }
        )
        df.index.name = "date"
        df.reset_index(inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        df.dropna(inplace=True)
        result[code] = df
    return result


def load_config(cfg_path: Path) -> list[dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: dict[str, Any]) -> tuple[str, Selector]:
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class", "")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    # 先尝试从 Selector 模块加载
    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError):
        # 如果 Selector 模块中没有，尝试从 ExtSelector 模块加载
        try:
            module = importlib.import_module("ExtSelector")
            cls = getattr(module, cls_name)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"无法加载 Selector.{cls_name} 或 ExtSelector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    parser.add_argument(
        "--stocklist",
        default="./stocklist.csv,./position.csv",
        help="股票池文件,可以指定多个文件，用逗号分隔",
    )
    parser.add_argument(
        "--exclude-boards",
        nargs="*",
        default=["gem", "star", "bj"],
        choices=["gem", "star", "bj"],
        help="排除板块，可多选：gem(创业板300/301) star(科创板688) bj(北交所.BJ/4/8)",
    )
    parser.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    parser.add_argument("--date", help="交易日 YYYY-MM-DD；缺省今天日期")
    parser.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    args = parser.parse_args()

    # --- 加载行情 --- #
    exclude_boards = set(args.exclude_boards or [])
    stocklist_paths = [Path(path) for path in args.stocklist.split(",")]
    stocklist_df = load_stocklist(stocklist_paths)
    stocklist_df = filter_by_boards_stocklist(stocklist_df, exclude_boards)
    codes = stocklist_df["ts_code"].tolist()
    trade_date: datetime = (
        pd.to_datetime(args.date)
        if args.date
        else datetime.combine(datetime.now(), datetime.min.time())
    )
    data = load_data(codes)
    if not args.date:
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        logger.info("%s", ", ".join(picks) if picks else "无符合条件股票")
        for pick in picks:
            target_df = stocklist_df[stocklist_df["ts_code"] == pick]
            logger.info("%s", target_df.to_string(index=False, header=False))


if __name__ == "__main__":
    main()
