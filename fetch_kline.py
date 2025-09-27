from __future__ import annotations

import argparse
import datetime as dt
import logging
import random
from typing import Literal, NamedTuple
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import akshare as ak
from mootdx.quotes import Quotes
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

# 屏蔽第三方库多余 INFO 日志
for noisy in ("httpx", "urllib3", "_client", "akshare"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

COLUMN_MAP_HIST_AK = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
}

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

FUND_CODE_PREFIX = NamedTuple(
    "FUND_CODE_PREFIX",
    [
        ("SH", tuple[str, ...]),
        ("SZ", tuple[str, ...]),
    ],
)(  # type: ignore[operator]
    SH=("50", "51", "52", "588"),
    SZ=("00", "15", "16"),
)

# --------------------------- 限流/封禁处理配置 --------------------------- #
COOLDOWN_SECS = 600
BAN_PATTERNS = (
    "访问频繁",
    "请稍后",
    "超过频率",
    "频繁访问",
    "too many requests",
    "429",
    "forbidden",
    "403",
    "max retries exceeded",
)


def _looks_like_ip_ban(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return any(pat in msg for pat in BAN_PATTERNS)


def _cool_sleep(base_seconds: int) -> None:
    jitter = random.uniform(0.9, 1.2)
    sleep_s = max(1, int(base_seconds * jitter))
    logger.warning("疑似被限流/封禁，进入冷却期 %d 秒...", sleep_s)
    time.sleep(sleep_s)


# ---------- AKShare 工具函数 ---------- #


def _get_kline_akshare(code: str, start: str, end: str) -> pd.DataFrame:
    fund_prefixes = FUND_CODE_PREFIX.SH + FUND_CODE_PREFIX.SZ
    for attempt in range(1, 4):
        try:
            if code.startswith(fund_prefixes):
                df = ak.fund_etf_hist_em(
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="qfq",
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=start,
                    end_date=end,
                    adjust="qfq",
                )
            break
        except Exception as e:
            logger.warning("AKShare 拉取 %s 失败(%d/3): %s", code, attempt, e)
            time.sleep(random.uniform(1, 2) * attempt)
    else:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = (
        df[list(COLUMN_MAP_HIST_AK)]
        .rename(columns=COLUMN_MAP_HIST_AK)
        .assign(date=lambda x: pd.to_datetime(x["date"]))
    )
    df[[c for c in df.columns if c != "date"]] = df[[c for c in df.columns if c != "date"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df = df[["date", "open", "close", "high", "low", "volume"]]
    return df.sort_values("date").reset_index(drop=True)


# ---------- Mootdx 工具函数 ---------- #


def _get_kline_mootdx(code: str, start: str, end: str) -> pd.DataFrame:
    symbol = code.zfill(6)
    freq = "day"
    client = Quotes.factory(market="std")
    try:
        df = client.bars(symbol=symbol, frequency=freq, adjust="qfq")
    except Exception as e:
        logger.warning("Mootdx 拉取 %s 失败: %s", code, e)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.rename(
        columns={
            "datetime": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "vol": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    start_ts = pd.to_datetime(start, format="%Y%m%d")
    end_ts = pd.to_datetime(end, format="%Y%m%d")
    df = df[(df["date"].dt.date >= start_ts.date()) & (df["date"].dt.date <= end_ts.date())].copy()
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "open", "close", "high", "low", "volume"]]


# ---------- 数据校验 ---------- #


def validate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
    if df["date"].isna().any():
        raise ValueError("存在缺失日期！")
    if (df["date"] > pd.Timestamp.today()).any():
        raise ValueError("数据包含未来日期，可能抓取错误！")
    return df


# --------------------------- 读取 stocklist.csv & 过滤板块 --------------------------- #


def _filter_by_boards_stocklist(df: pd.DataFrame, exclude_boards: set[str]) -> pd.DataFrame:
    """
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


def load_codes_from_stocklist(stocklist_csv: Path, exclude_boards: set[str]) -> list[str]:
    df = pd.read_csv(stocklist_csv)
    df = _filter_by_boards_stocklist(df, exclude_boards)
    codes = df["symbol"].astype(str).str.zfill(6).tolist()
    codes = list(dict.fromkeys(codes))  # 去重保持顺序
    logger.info(
        "从 %s 读取到 %d 只股票（排除板块：%s）",
        stocklist_csv,
        len(codes),
        ",".join(sorted(exclude_boards)) or "无",
    )
    return codes


# --------------------------- 单只抓取（全量覆盖保存） --------------------------- #


def fetch_one(
    code: str,
    start: str,
    end: str,
    out_dir: Path,
    *,
    data_source: Literal["akshare", "mootdx"] = "akshare",
) -> None:
    csv_path = out_dir / f"{code}.csv"

    for attempt in range(1, 4):
        try:
            match data_source:
                case "akshare":
                    new_df = _get_kline_akshare(code, start, end)
                case "mootdx":
                    new_df = _get_kline_mootdx(code, start, end)
                case _:
                    raise ValueError(f"不支持的数据源: {data_source}")
            if new_df.empty:
                logger.debug("%s 无数据，生成空表。", code)
                new_df = pd.DataFrame(columns=["date", "open", "close", "high", "low", "volume"])
            new_df = validate(new_df)
            new_df.to_csv(csv_path, index=False)  # 直接覆盖保存
            break
        except Exception as e:
            if _looks_like_ip_ban(e):
                logger.error(f"{code} 第 {attempt} 次抓取疑似被封禁，沉睡 {COOLDOWN_SECS} 秒")
                _cool_sleep(COOLDOWN_SECS)
            else:
                silent_seconds = 15 * attempt
                logger.info(f"{code} 第 {attempt} 次抓取失败，{silent_seconds} 秒后重试：{e}")
                time.sleep(silent_seconds)
    else:
        logger.error("%s 三次抓取均失败，已跳过！", code)


# ---------- 主入口 ---------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="按市值筛选 A 股并抓取历史 K 线")
    parser.add_argument("--start", default="20190101", help="起始日期 YYYYMMDD 或 'today'")
    parser.add_argument("--end", default="today", help="结束日期 YYYYMMDD 或 'today'")
    # 股票清单与板块过滤
    parser.add_argument(
        "--stocklist",
        type=Path,
        default=Path("./stocklist.csv"),
        help="股票清单CSV路径（需含 ts_code 或 symbol）",
    )
    parser.add_argument(
        "--exclude-boards",
        nargs="*",
        default=["gem", "star", "bj"],
        choices=["gem", "star", "bj"],
        help="排除板块，可多选：gem(创业板300/301) star(科创板688) bj(北交所.BJ/4/8)",
    )
    # 其它
    parser.add_argument("--out", default="./data", help="输出目录")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数")
    parser.add_argument(
        "--data-source", type=str, default="akshare", help="数据源：akshare(akshare) mootdx(mootdx)"
    )
    args = parser.parse_args()

    # ---------- 日期解析 ---------- #
    start = dt.date.today().strftime("%Y%m%d") if str(args.start).lower() == "today" else args.start
    end = dt.date.today().strftime("%Y%m%d") if str(args.end).lower() == "today" else args.end

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 从 stocklist.csv 读取股票池 ---------- #
    exclude_boards = set(args.exclude_boards or [])
    codes = load_codes_from_stocklist(args.stocklist, exclude_boards)

    if not codes:
        logger.error("stocklist 为空或被过滤后无代码，请检查。")
        sys.exit(1)

    logger.info(
        "开始抓取 %d 支股票 | 数据源:%s(日线,qfq) | 日期:%s → %s | 排除:%s",
        len(codes),
        args.data_source,
        start,
        end,
        ",".join(sorted(exclude_boards)) or "无",
    )

    # ---------- 多线程抓取（全量覆盖） ---------- #
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                fetch_one,
                code,
                start,
                end,
                out_dir,
                data_source=args.data_source,
            )
            for code in codes
        ]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="下载进度"):
            pass

    logger.info("全部任务完成，数据已保存至 %s", out_dir.resolve())


if __name__ == "__main__":
    main()
