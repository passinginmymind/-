#!/usr/bin/env python3
"""
기관 3일 연속 순매수 종목 백테스트

전략
  - 신호: 기관(기관합계) 순매수거래대금이 N일(기본 3일) 연속 양수인 종목
  - 진입: 신호 발생 다음 거래일 시가에 매수
  - 청산: 진입일 포함 H거래일(기본 5일) 보유 후 H번째 거래일 종가에 매도
  - 대상: KOSPI + KOSDAQ 전 종목, 기간: 최근 3년

데이터: pykrx (KRX 정보데이터시스템)
  - 일자별 전 종목 OHLCV         : stock.get_market_ohlcv(date, market=...)
  - 일자별 전 종목 기관 순매수     : stock.get_market_net_purchases_of_equities(date, date, market, "기관합계")
  하루 단위로 받아 ./cache 에 CSV 로 저장하므로 두 번째 실행부터는 빠릅니다.

수익률 계산
  액면분할/병합 등으로 원시 가격이 불연속이 되는 문제를 피하기 위해
  (진입일 종가 / 진입일 시가) x Π(1 + 이후 일별 등락률) 로 계산합니다.
  KRX 등락률은 권리락 등이 반영된 기준가 대비 수치입니다.

사용 예
  pip install pykrx pandas numpy
  python institutional_backtest.py
  python institutional_backtest.py --years 3 --streak 3 --hold 5 --min-value 1e9
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

try:
    from pykrx import stock
except ImportError:  # pragma: no cover
    sys.exit("pykrx 가 필요합니다:  pip install pykrx")


MARKETS = ("KOSPI", "KOSDAQ")
INVESTOR = "기관합계"


# --------------------------------------------------------------------------- #
# 데이터 수집 (일자 단위 캐시)
# --------------------------------------------------------------------------- #
def _retry(func, *args, retries: int = 3, wait: float = 1.0, **kwargs):
    for i in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # 네트워크 오류 등
            if i == retries - 1:
                raise
            print(f"  재시도 {i + 1}/{retries}: {e}", file=sys.stderr)
            time.sleep(wait * (2 ** i))


def get_trading_days(start: str, end: str) -> list[str]:
    """삼성전자 일봉 인덱스로 거래일 목록을 구한다."""
    df = _retry(stock.get_market_ohlcv, start, end, "005930")
    return [d.strftime("%Y%m%d") for d in df.index]


def load_day(date: str, cache_dir: str, sleep: float) -> pd.DataFrame | None:
    """해당 일자의 전 종목 OHLCV + 기관 순매수대금을 반환 (index=티커)."""
    path = os.path.join(cache_dir, f"{date}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, dtype={"ticker": str}).set_index("ticker")

    frames = []
    for mkt in MARKETS:
        ohlcv = _retry(stock.get_market_ohlcv, date, market=mkt)
        time.sleep(sleep)
        net = _retry(stock.get_market_net_purchases_of_equities, date, date, mkt, INVESTOR)
        time.sleep(sleep)
        if ohlcv is None or ohlcv.empty:
            continue

        df = pd.DataFrame(
            {
                "open": ohlcv["시가"],
                "close": ohlcv["종가"],
                "value": ohlcv["거래대금"],
                "chg": ohlcv["등락률"],
            }
        )
        if net is not None and not net.empty:
            df["inst_net"] = net["순매수거래대금"].reindex(df.index)
        else:
            df["inst_net"] = np.nan
        df["market"] = mkt
        frames.append(df)

    if not frames:
        return None  # 장 마감 전/휴장 등으로 데이터 없음 (캐시하지 않음)

    out = pd.concat(frames)
    out.index.name = "ticker"
    out = out[~out.index.duplicated()]
    out.reset_index().to_csv(path, index=False)
    return out


def build_panel(days: list[str], cache_dir: str, sleep: float) -> dict[str, pd.DataFrame]:
    """필드별 (날짜 x 티커) 패널을 만든다."""
    os.makedirs(cache_dir, exist_ok=True)
    per_day = {}
    for i, d in enumerate(days, 1):
        cached = os.path.exists(os.path.join(cache_dir, f"{d}.csv"))
        if not cached or i % 50 == 0 or i == len(days):
            print(f"[{i}/{len(days)}] {d}{' (cache)' if cached else ''}")
        df = load_day(d, cache_dir, sleep)
        if df is None:
            print(f"  {d}: 데이터 없음 → 제외", file=sys.stderr)
            continue
        per_day[d] = df

    if not per_day:
        sys.exit("데이터를 받지 못했습니다. KRX 접속 상태와 pykrx 버전을 확인하세요 (pip install -U pykrx).")

    panel = {}
    for field in ("open", "close", "value", "chg", "inst_net"):
        panel[field] = pd.DataFrame({d: df[field] for d, df in per_day.items()}).T.sort_index()
    return panel


# --------------------------------------------------------------------------- #
# 백테스트
# --------------------------------------------------------------------------- #
def run_backtest(
    panel: dict[str, pd.DataFrame],
    bt_days: list[str],
    streak: int,
    hold: int,
    min_value: float,
    fee: float,
    tax: float,
    allow_overlap: bool,
) -> pd.DataFrame:
    days = list(panel["open"].index)
    pos = {d: i for i, d in enumerate(days)}
    opn, cls, val, chg, net = (panel[k] for k in ("open", "close", "value", "chg", "inst_net"))

    # N일 연속 순매수 여부 (NaN 은 순매수 아님으로 처리)
    buy = (net > 0).astype(int)
    signal = buy.rolling(streak).sum() == streak

    trades = []
    busy_until: dict[str, int] = {}  # 티커별 보유 종료 인덱스 (중복 진입 방지)

    for d in bt_days:
        if d not in pos:
            continue
        i = pos[d]
        entry_i = i + 1
        if i < streak - 1 or entry_i >= len(days):
            continue
        exit_i = entry_i + hold - 1
        if exit_i >= len(days):
            continue  # 보유 기간이 데이터 끝을 넘으면 제외

        tickers = signal.columns[signal.iloc[i].values]
        entry_day = days[entry_i]

        for t in tickers:
            if not allow_overlap and busy_until.get(t, -1) >= entry_i:
                continue
            if min_value > 0 and not (val.at[d, t] >= min_value):
                continue

            o = opn.at[entry_day, t]
            c = cls.at[entry_day, t]
            if not (o > 0 and c > 0):  # 거래정지/데이터 없음
                continue

            # 진입일 종가/시가 x 이후 일별 등락률 누적
            gross = c / o
            last_i = entry_i
            for k in range(entry_i + 1, exit_i + 1):
                r = chg.iat[k, chg.columns.get_loc(t)]
                ck = cls.iat[k, cls.columns.get_loc(t)]
                if pd.isna(r) or not (ck > 0):
                    break  # 상장폐지/정지: 마지막 가용 종가에 청산 처리
                gross *= 1 + r / 100.0
                last_i = k

            net_ret = gross * (1 - fee) * (1 - fee - tax) - 1
            trades.append(
                {
                    "signal_date": d,
                    "entry_date": entry_day,
                    "exit_date": days[last_i],
                    "ticker": t,
                    "entry_price": o,
                    "gross_return": gross - 1,
                    "return": net_ret,
                    "held_days": last_i - entry_i + 1,
                }
            )
            busy_until[t] = exit_i

    return pd.DataFrame(trades)


# --------------------------------------------------------------------------- #
# 리포트
# --------------------------------------------------------------------------- #
def summarize(r: pd.Series) -> dict:
    wins, losses = r[r > 0], r[r <= 0]
    pf = wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf
    return {
        "거래수": len(r),
        "승률(%)": 100 * (r > 0).mean(),
        "평균수익률(%)": 100 * r.mean(),
        "중앙값(%)": 100 * r.median(),
        "평균이익(%)": 100 * wins.mean() if len(wins) else np.nan,
        "평균손실(%)": 100 * losses.mean() if len(losses) else np.nan,
        "손익비(PF)": pf,
        "최대이익(%)": 100 * r.max(),
        "최대손실(%)": 100 * r.min(),
    }


def report(trades: pd.DataFrame, hold: int) -> None:
    if trades.empty:
        print("거래가 없습니다.")
        return

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    pd.set_option("display.width", 160)

    print("\n=== 전체 성과 (비용 차감 후) ===")
    for k, v in summarize(trades["return"]).items():
        print(f"  {k:<12}: {v:,.2f}" if isinstance(v, float) else f"  {k:<12}: {v:,}")
    print(f"  (비용 차감 전 평균수익률: {100 * trades['gross_return'].mean():.2f}%, "
          f"승률: {100 * (trades['gross_return'] > 0).mean():.2f}%)")

    print("\n=== 연도별 ===")
    by_year = trades.groupby(trades["entry_date"].str[:4])["return"].apply(
        lambda s: pd.Series(summarize(s))
    ).unstack()
    print(by_year[["거래수", "승률(%)", "평균수익률(%)", "중앙값(%)", "손익비(PF)"]])

    # 신호일별 동일가중 바스켓 → 자본을 hold 개 슬리브로 나눠 매일 한 슬리브씩 투입했다고 가정
    daily = trades.groupby("entry_date")["return"].mean()
    sleeve_curve = (1 + daily / hold).cumprod()
    print("\n=== 포트폴리오 근사 (자본 1/{0}씩 매일 신규 바스켓 동일가중 투입) ===".format(hold))
    print(f"  진입일 수          : {len(daily):,}")
    print(f"  바스켓 승률(%)     : {100 * (daily > 0).mean():.2f}")
    print(f"  바스켓 평균수익(%) : {100 * daily.mean():.2f}")
    print(f"  누적수익률(%)      : {100 * (sleeve_curve.iloc[-1] - 1):.2f}")
    mdd = (sleeve_curve / sleeve_curve.cummax() - 1).min()
    print(f"  MDD(%)             : {100 * mdd:.2f}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="기관 N일 연속 순매수 → 익일 시가 매수, H일 보유 백테스트")
    p.add_argument("--years", type=float, default=3, help="백테스트 기간(년), 기본 3")
    p.add_argument("--end", default=None, help="종료일 YYYYMMDD (기본: 어제)")
    p.add_argument("--streak", type=int, default=3, help="연속 순매수 일수, 기본 3")
    p.add_argument("--hold", type=int, default=5, help="보유 거래일 수(진입일 포함), 기본 5")
    p.add_argument("--min-value", type=float, default=0,
                   help="신호일 최소 거래대금(원). 예: 1e9 = 10억. 기본 0(필터 없음)")
    p.add_argument("--fee", type=float, default=0.00015, help="편도 수수료율, 기본 0.015%%")
    p.add_argument("--tax", type=float, default=0.0020, help="매도 거래세율, 기본 0.20%%")
    p.add_argument("--allow-overlap", action="store_true",
                   help="이미 보유 중인 종목에 새 신호가 나도 중복 진입 허용")
    p.add_argument("--cache-dir", default="cache", help="일자별 데이터 캐시 폴더")
    p.add_argument("--sleep", type=float, default=0.3, help="KRX 요청 간 대기(초)")
    p.add_argument("--out", default="trades.csv", help="거래 내역 CSV 경로")
    a = p.parse_args()

    end_dt = datetime.strptime(a.end, "%Y%m%d") if a.end else datetime.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=int(365 * a.years))
    # 연속 순매수 판정용 여유 구간 (앞쪽)
    warmup_dt = start_dt - timedelta(days=a.streak * 3 + 10)

    start, end, warmup = (x.strftime("%Y%m%d") for x in (start_dt, end_dt, warmup_dt))
    print(f"기간 {start} ~ {end} | 기관 {a.streak}일 연속 순매수 → 익일 시가 매수, {a.hold}일 보유")

    all_days = get_trading_days(warmup, end)
    bt_days = [d for d in all_days if d >= start]
    print(f"거래일 {len(all_days)}일 (백테스트 {len(bt_days)}일) 데이터 로딩...")

    panel = build_panel(all_days, a.cache_dir, a.sleep)
    trades = run_backtest(
        panel, bt_days, a.streak, a.hold, a.min_value, a.fee, a.tax, a.allow_overlap
    )

    report(trades, a.hold)
    if not trades.empty:
        trades.to_csv(a.out, index=False, encoding="utf-8-sig")
        print(f"\n거래 내역 저장: {a.out}")


if __name__ == "__main__":
    main()
