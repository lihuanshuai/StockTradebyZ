from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

import pandas as pd
from xtquant import xtdata

# --------------------------- 全局日志配置 --------------------------- #
LOG_FILE = Path("fetch.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

STOCK_CODE_PREFIX = NamedTuple(
    "STOCK_CODE_PREFIX",
    [
        ("GEM", tuple[str, ...]),
        ("STAR", tuple[str, ...]),
        ("BJ", tuple[str, ...]),
    ],
)(  # type: ignore[operator]
    GEM=("300", "301"),
    STAR=("688",),
    BJ=("4", "8"),
)


# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #


def filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    """
    过滤板块
    exclude_boards 子集：{'gem','star','bj'}
    - gem  : 创业板 300/301（.SZ）
    - star : 科创板 688（.SH）
    - bj   : 北交所（.BJ 或 4/8 开头）
    """
    code = df["symbol"].astype(str)
    ts_code = df["ts_code"].astype(str).str.upper()
    mask = pd.Series(True, index=df.index)

    if "gem" in exclude_boards:
        mask &= ~code.str.startswith(STOCK_CODE_PREFIX.GEM)
    if "star" in exclude_boards:
        mask &= ~code.str.startswith(STOCK_CODE_PREFIX.STAR)
    if "bj" in exclude_boards:
        mask &= ~(ts_code.str.endswith(".BJ") | code.str.startswith(STOCK_CODE_PREFIX.BJ))

    return df[mask].copy()


def load_stocklist(stocklist_paths: list[Path]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(path) for path in stocklist_paths])
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df = df.drop_duplicates(subset=["ts_code"])
    return df


# ---------- 主入口 ---------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--start", default="", help="起始日期 YYYYMMDD 或 ''")
    # 股票清单与板块过滤
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
    # 其它
    args = parser.parse_args()

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    exclude_boards = set(args.exclude_boards or [])
    stocklist_paths = [Path(path) for path in args.stocklist.split(",")]
    stocklist_df = load_stocklist(stocklist_paths)
    stocklist_df = filter_by_boards_stocklist(stocklist_df, exclude_boards)
    codes = stocklist_df["ts_code"].tolist()
    logger.info(
        "从 %s 读取到 %d 只股票（排除板块：%s）",
        ",".join([path.name for path in stocklist_paths]),
        len(codes),
        ",".join(sorted(exclude_boards)) or "无",
    )

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:xtdata(日线,qfq) | 排除:%s",
        len(codes),
        ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 抓取数据 ---------- #
    def callback(data: dict[str, Any], prefix: str = "") -> None:
        print(f"{prefix}: {data}")

    # 下载日线数据
    logger.info("开始下载日线数据")
    xtdata.download_history_data2(
        codes, "1d", start_time=args.start, callback=partial(callback, prefix="日线")
    )
    # 下载财务数据
    logger.info("开始下载财务数据")
    xtdata.download_financial_data2(
        codes,
        table_list=[],
        start_time=args.start,
        callback=partial(callback, prefix="财务"),
    )
    logger.info("开始下载5分钟线数据")
    xtdata.download_history_data2(
        codes, "5m", start_time=args.start, callback=partial(callback, prefix="5分钟线")
    )
    # # 下载行业数据
    # logger.info("开始下载行业数据")
    # xtdata.download_sector_data()

    logger.info("全部任务完成，数据已保存")


if __name__ == "__main__":
    main()
