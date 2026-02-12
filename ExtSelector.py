import logging

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from Selector import Selector, compute_kdj, passes_day_constraints_today, zx_condition_at_positions

logger = logging.getLogger("select")


def compute_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    计算 CMF (Chaikin Money Flow) 指标

    公式：
    - MF_UP = CLOSE - LOW
    - MF_DOWN = HIGH - CLOSE
    - FACTOR = 1 / (HIGH - LOW) if HIGH - LOW > 0 else 0
    - MFM = (MF_UP - MF_DOWN) * FACTOR
    - MF_VOL = MFM * VOL
    - CMF = SUM(MF_VOL, N) / SUM(VOL, N)

    Parameters
    ----------
    df : pd.DataFrame
        包含 high, low, close, volume 列的 DataFrame
    period : int, default 20
        CMF 计算周期

    Returns
    -------
    pd.Series
        CMF 指标序列
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float)

    # 计算资金流乘数 (Money Flow Multiplier)
    mf_up = close - low
    mf_down = high - close
    high_low_range = high - low

    # 避免除零
    factor = np.where(high_low_range > 0, 1.0 / high_low_range, 0.0)
    mfm = (mf_up - mf_down) * factor

    # 资金流体积
    mf_vol = mfm * volume

    # CMF = SUM(MF_VOL, N) / SUM(VOL, N)
    sum_mf_vol = pd.Series(mf_vol, index=df.index).rolling(window=period, min_periods=period).sum()
    sum_vol = volume.rolling(window=period, min_periods=period).sum()

    cmf = np.where(sum_vol > 0, sum_mf_vol / sum_vol, 0.0)

    return pd.Series(cmf, index=df.index)


class CMFDivergenceSelector(Selector):
    """
    CMF 底背离选股器

    检测逻辑：
    1. 在最近 lookback_n 个交易日内，找到价格的两个低点（t1 < t2，t2 是更近的低点）
    2. 检测是否存在底背离：
       - 价格在 t2 处创新低（或接近新低）：close[t2] <= close[t1] * (1 + price_tolerance)
       - CMF 在 t2 处高于 t1：cmf[t2] > cmf[t1]
    3. 其他过滤条件：
       - 当日约束（涨跌幅和振幅限制）
       - KDJ J 值条件（可选）
       - 知行条件（可选）
    """

    def __init__(
        self,
        *,
        cmf_period: int = 20,
        lookback_n: int = 60,
        price_tolerance: float = 0.02,  # 价格低点允许的偏差（2%）
        min_peak_distance: int = 10,  # 两个低点之间的最小距离
        min_cmf_improvement: float = 0.05,  # CMF 改善的最小幅度
        j_threshold: float | None = None,  # J 值阈值，None 表示不检查
        j_q_threshold: float | None = None,  # J 值分位阈值，None 表示不检查
        max_window: int = 120,  # 用于计算 J 分位的窗口
        require_zx_condition: bool = True,  # 是否要求知行条件
    ) -> None:
        if lookback_n < min_peak_distance * 2:
            raise ValueError(
                f"lookback_n ({lookback_n}) 应 >= min_peak_distance * 2 ({min_peak_distance * 2})"
            )
        if not (0 <= price_tolerance < 1):
            raise ValueError("price_tolerance 应位于 [0, 1) 区间")
        if min_cmf_improvement < 0:
            raise ValueError("min_cmf_improvement 应 >= 0")

        self.cmf_period = cmf_period
        self.lookback_n = lookback_n
        self.price_tolerance = price_tolerance
        self.min_peak_distance = min_peak_distance
        self.min_cmf_improvement = min_cmf_improvement
        self.j_threshold = j_threshold
        self.j_q_threshold = j_q_threshold
        self.max_window = max_window
        self.require_zx_condition = require_zx_condition

    def _find_price_lows(
        self,
        hist: pd.DataFrame,
        lookback_window: int,
    ) -> list[int]:
        """
        找到价格的低点位置（使用 find_peaks 在负价格序列上找）

        Returns
        -------
        list[int]
            低点的 iloc 位置列表（按时间顺序，从早到晚）
        """
        if len(hist) < lookback_window:
            return []

        # 取最近 lookback_window 个交易日
        window = hist.tail(lookback_window)
        close_prices = window["close"].values

        # 使用负价格找低点（find_peaks 找峰值）
        neg_prices = -close_prices

        # 找到低点
        indices, _ = find_peaks(
            neg_prices,
            distance=self.min_peak_distance,
            prominence=close_prices.std() * 0.5,  # 至少要有一定的显著性
        )

        # 转换为原始 DataFrame 的 iloc 位置
        window_start = len(hist) - lookback_window
        low_positions = [window_start + idx for idx in indices]

        return sorted(low_positions)

    def _detect_divergence(
        self,
        hist: pd.DataFrame,
        cmf: pd.Series,
    ) -> bool:
        """
        检测是否存在 CMF 底背离

        Returns
        -------
        bool
            是否存在底背离
        """
        # 找到价格低点
        low_positions = self._find_price_lows(hist, self.lookback_n)

        if len(low_positions) < 2:
            return False

        # 取最近的两个低点（t1 较早，t2 较晚）
        t1_pos = low_positions[-2]
        t2_pos = low_positions[-1]

        # 确保 t2 是最近的低点（接近当前日期）
        if t2_pos < len(hist) - 5:  # 如果最近的低点距离今天超过5天，可能不够新鲜
            return False

        # 获取价格和 CMF 值
        close_t1 = float(hist["close"].iloc[t1_pos])
        close_t2 = float(hist["close"].iloc[t2_pos])
        cmf_t1 = float(cmf.iloc[t1_pos])
        cmf_t2 = float(cmf.iloc[t2_pos])

        # 检查数据有效性
        if not (
            np.isfinite(close_t1)
            and np.isfinite(close_t2)
            and np.isfinite(cmf_t1)
            and np.isfinite(cmf_t2)
        ):
            return False

        if close_t1 <= 0:
            return False

        # 1. 价格条件：t2 处的价格 <= t1 处的价格 * (1 + tolerance)
        #    即价格创新低（或接近新低）
        price_condition = close_t2 <= close_t1 * (1 + self.price_tolerance)

        if not price_condition:
            return False

        # 2. CMF 条件：t2 处的 CMF > t1 处的 CMF + min_improvement
        #    即 CMF 改善（底背离）
        cmf_condition = cmf_t2 > cmf_t1 + self.min_cmf_improvement

        if cmf_t2 > cmf_t1:
            price_tolerance = close_t2 / close_t1 - 1
            cmf_improvement = cmf_t2 - cmf_t1
            logger.debug(
                f"CMF 底背离检测：price_tolerance={price_tolerance:.4f}, cmf_improvement={cmf_improvement:.4f}"
            )

        return cmf_condition

    def _passes_filters(self, hist: pd.DataFrame) -> bool:
        """
        单支股票的过滤逻辑
        """
        if hist.empty:
            return False

        hist = hist.copy().sort_values("date")

        # 数据量检查
        min_len = max(self.cmf_period + self.lookback_n, self.max_window + 20)
        if len(hist) < min_len:
            return False

        # 当日约束
        if not passes_day_constraints_today(hist):
            return False

        # 计算 CMF
        cmf = compute_cmf(hist, period=self.cmf_period)

        # 检测底背离
        if not self._detect_divergence(hist, cmf):
            return False

        # KDJ J 值条件（可选）
        if self.j_threshold is not None or self.j_q_threshold is not None:
            kdj = compute_kdj(hist)
            j_today = float(kdj["J"].iloc[-1])

            if self.j_threshold is not None and j_today >= self.j_threshold:
                return False

            if self.j_q_threshold is not None:
                j_window = kdj["J"].tail(self.max_window).dropna()
                if not j_window.empty:
                    j_q_val = float(j_window.quantile(self.j_q_threshold))
                    if j_today > j_q_val:
                        return False

        # 知行条件（可选）
        if self.require_zx_condition:
            if not zx_condition_at_positions(
                hist,
                require_close_gt_long=True,
                require_short_gt_long=True,
                pos=None,
            ):
                return False

        return True

    def select(self, date: pd.Timestamp, data: dict[str, pd.DataFrame]) -> list[str]:
        """
        批量选股接口
        """
        picks: list[str] = []
        min_len = max(self.cmf_period + self.lookback_n, self.max_window + 20)

        for code, df in data.items():
            hist = df[df["date"] <= date].tail(min_len)
            if len(hist) < min_len:
                continue
            if self._passes_filters(hist):
                picks.append(code)

        return picks
