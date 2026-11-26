import os
import json
import shutil
import tempfile

# 修复 Windows 中文路径下 curl_cffi 无法读取 certifi/cacert.pem (curl error 77)
try:
    import certifi

    _ca_path = certifi.where()
    if not _ca_path.isascii():
        _ascii_ca = os.path.join(tempfile.gettempdir(), "quant_cacert.pem")
        if not os.path.exists(_ascii_ca) or os.path.getsize(_ascii_ca) != os.path.getsize(_ca_path):
            shutil.copy2(_ca_path, _ascii_ca)
        os.environ["CURL_CA_BUNDLE"] = _ascii_ca
        os.environ["SSL_CERT_FILE"] = _ascii_ca
        os.environ["REQUESTS_CA_BUNDLE"] = _ascii_ca
except Exception:
    pass

import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone

CONFIG = {
    "model_version": "v2.0",
    "start_date": "1999-03-10",
    "max_gross": 1.50,          # Gross 上限 150%
    "min_gross": 0.60,          # 极端高波下保留最低 Gross 60%
    "max_net": 1.20,            # 强多头最高 Net +120% (如 Long 135% / Short 15%)
    "min_net": -0.25,           # 确认熊市最低 Net -25%
    "target_vol_gross": 0.28,   # Gross 波动率目标 28%（适配科技/AI 弹性）
    "ema_span": 12,             # 综合得分 12 日 EMA 平滑，过滤日内噪音与假突破
    "rebalance_buffer": 0.15,   # 15% Net 调仓缓冲区，大幅压低换手磨损
    "bear_net_cap": 0.05,       # 长期熊市门控（P<MA200 且 MA50<MA200）Net 封顶 +5%
    "bull_net_floor": 0.85,     # 长期牛市门控（趋势+相对强度+低波共振）Net 保底 +85%
    "vix_capitulation": 40.0,   # 极端恐慌（VIX>40）禁止保留/追空净空头，防暴力轧空
    "margin_spread": 0.015,     # 杠杆融资点差（无风险利率 + 1.5%）
    "tx_cost": 0.0005,          # 单边交易滑点与冲击成本 5bp
}

TICKERS = ["QQQ", "^SOX", "^NDX", "^GSPC", "VWEHX", "VFITX", "^TNX", "^IRX", "^VIX"]


def map_net(s: float) -> float:
    if s >= 0.30:
        return 0.95 + (s - 0.30) / 0.70 * (CONFIG["max_net"] - 0.95)
    elif s >= 0.00:
        return 0.50 + (s / 0.30) * 0.45
    elif s >= -0.35:
        return 0.00 + ((s + 0.35) / 0.35) * 0.50
    else:
        return CONFIG["min_net"] + ((s + 1.0) / 0.65) * (0.00 - CONFIG["min_net"])


def get_regime(s: float) -> dict:
    if s >= 0.30:
        return {
            "label": "强多头（主升浪）",
            "color": "#10b981",
            "action": "高净多满格进攻，容忍常规洗盘，保留少量高确定性空头 Alpha",
        }
    elif s >= 0.00:
        return {
            "label": "温和偏多 / 震荡蓄势",
            "color": "#38bdf8",
            "action": "保持显著偏多配置，精选 AI/科技多空个股，不因短期噪音频繁折腾",
        }
    elif s >= -0.35:
        return {
            "label": "中性防御 / 趋势转弱",
            "color": "#f59e0b",
            "action": "用 QQQ/MNQ 指数空头对冲多余 Beta，降低杠杆、聚焦纯 Alpha",
        }
    else:
        return {
            "label": "确认熊市 / 系统性风险",
            "color": "#ef4444",
            "action": "允许适度净空保护净值并严控 Gross；若 VIX>40 极端恐慌则不追空防逼空",
        }


def calc_metrics(r: pd.Series, rf: pd.Series) -> dict:
    cum = (1 + r).cumprod()
    n_years = len(r) / 252.0
    cagr = float(cum.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else 0.0
    vol = float(r.std() * np.sqrt(252))
    rf_ann = float(rf.loc[r.index].mean() * 252)
    sharpe = float((r.mean() * 252 - rf_ann) / vol) if vol > 0 else 0.0
    dd = cum / cum.cummax() - 1
    mdd = float(dd.min())
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {
        "cagr": round(cagr, 4),
        "vol": round(vol, 4),
        "sharpe": round(sharpe, 2),
        "mdd": round(mdd, 4),
        "calmar": round(calmar, 2),
    }


def extract_major_rebalances(
    bt: pd.DataFrame, threshold: float = 0.25, cooldown_days: int = 20
) -> list:
    """
    扫描 5 日滚动 Net 变动超过 threshold (默认 25%) 的重大加仓/减仓事件，
    并设置 cooldown_days 避免同一波行情里连续几天重复记录。
    """
    net_5d_chg = bt["exec_net"] - bt["exec_net"].shift(5)
    events = []
    last_idx = -cooldown_days

    for i in range(5, len(bt)):
        chg = net_5d_chg.iloc[i]
        if pd.notna(chg) and abs(chg) >= threshold and (i - last_idx) >= cooldown_days:
            row = bt.iloc[i]
            prev_row = bt.iloc[i - 5]

            factor_contrib = {
                "趋势动量": 0.45 * row["score_trend"],
                "科技相对强弱": 0.25 * row["score_rel"],
                "波动率体制": 0.15 * row["score_volreg"],
                "信用与利率": 0.15 * row["score_macro"],
            }
            driver = max(factor_contrib, key=lambda k: abs(factor_contrib[k]))

            events.append(
                {
                    "date": bt.index[i].strftime("%Y-%m-%d"),
                    "action": "重大加仓" if chg > 0 else "重大减仓",
                    "net_before": round(float(prev_row["exec_net"]), 3),
                    "net_after": round(float(row["exec_net"]), 3),
                    "delta_net": round(float(chg), 3),
                    "gross_after": round(float(row["target_gross"]), 3),
                    "score_S": round(float(row["S"]), 2),
                    "main_driver": driver,
                }
            )
            last_idx = i

    return events


def download_close_data(tickers: list, start_date: str) -> pd.DataFrame:
    raw = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance 未返回任何行情数据，请检查网络连接。")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        else:
            raise KeyError(f"下载数据中缺少 Close 字段，现有字段: {raw.columns.levels[0].tolist()}")
    else:
        close = raw[["Close"]].copy() if "Close" in raw.columns else raw.copy()

    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    for t in missing:
        single = yf.download(t, start=start_date, auto_adjust=True, progress=False)
        if single is not None and not single.empty:
            s_close = single["Close"]
            if isinstance(s_close, pd.DataFrame):
                s_close = s_close.iloc[:, 0]
            close[t] = s_close

    still_missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    if still_missing:
        raise RuntimeError(f"以下标的数据下载失败: {still_missing}")

    close = close[tickers].sort_index()
    print("=== 各标的数据覆盖范围 ===")
    for t in tickers:
        s = close[t].dropna()
        print(f"  {t:6s}: {s.index[0].strftime('%Y-%m-%d')} ~ {s.index[-1].strftime('%Y-%m-%d')} ({len(s)} rows)")

    # 严格以 QQQ 真实美股交易日历为基准，避免假期 VIX 单独报价产生虚拟交易日
    qqq_dates = close["QQQ"].dropna().index
    close = close.ffill().loc[qqq_dates].dropna()
    return close


def send_notifications(payload: dict):
    latest = payload["latest"]
    gates = latest.get("gates", {})
    active_gates = [k for k, v in [
        ("牛市底仓保护", gates.get("secular_bull")),
        ("熊市防诱多封顶", gates.get("secular_bear")),
        ("VIX>40平空保护", gates.get("vix_capitulation")),
    ] if v]
    gate_str = " / ".join(active_gates) if active_gates else "无触发"

    msg = (
        f"📊 科技/AI 量化敞口日报 v2.0 ({latest['date']})\n"
        f"状态：{latest['regime']['label']}\n"
        f"综合得分 S：{latest['S']:+.2f}（门控：{gate_str}）\n"
        f"• 长期趋势: {latest['scores']['trend']:+.2f} | 相对标普: {latest['scores']['rel_spx']:+.2f}\n"
        f"• 波动体制: {latest['scores']['vol_regime']:+.2f} | 信用利率: {latest['scores']['macro']:+.2f}\n"
        f"——————————————\n"
        f"🎯 Target Net：{latest['exec_net']:+.1%}（原始 {latest['target_net']:+.1%}）\n"
        f"⚖️ Target Gross：{latest['target_gross']:.1%}\n"
        f"📌 参考配比：Long {latest['target_long']:.1%} / Short {latest['target_short']:.1%}\n"
        f"💡 建议：{latest['regime']['action']}"
    )

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID")
    if tg_token and tg_chat:
        try:
            requests.post(
                f"https://api.telegram.org/bot{tg_token}/sendMessage",
                json={"chat_id": tg_chat, "text": msg},
                timeout=10,
            )
            print("Telegram notification sent.")
        except Exception as e:
            print(f"Telegram push failed: {e}")

    feishu_url = os.getenv("FEISHU_WEBHOOK_URL")
    if feishu_url:
        try:
            requests.post(
                feishu_url,
                json={"msg_type": "text", "content": {"text": msg}},
                timeout=10,
            )
            print("Feishu notification sent.")
        except Exception as e:
            print(f"Feishu push failed: {e}")


def main():
    close = download_close_data(TICKERS, CONFIG["start_date"])

    ret_qqq = close["QQQ"].pct_change().fillna(0)
    ret_sox = close["^SOX"].pct_change().fillna(0)
    ret_bench = 0.70 * ret_qqq + 0.30 * ret_sox
    bench_idx = (1 + ret_bench).cumprod()
    rf_daily = (close["^IRX"] / 100.0).clip(lower=0) / 252.0
    spx = close["^GSPC"]

    df = pd.DataFrame(index=close.index)

    # 基础技术序列（严格无未来函数）
    ma50 = bench_idx.rolling(50).mean()
    ma100 = bench_idx.rolling(100).mean()
    ma200 = bench_idx.rolling(200).mean()
    rv20 = ret_bench.rolling(20).std() * np.sqrt(252)
    rv60 = ret_bench.rolling(60).std() * np.sqrt(252)

    credit_ratio = close["VWEHX"] / close["VFITX"]
    cr_ma100 = credit_ratio.rolling(100).mean()
    cr_ma150 = credit_ratio.rolling(150).mean()

    rel_spx = bench_idx / spx
    rel_ma100 = rel_spx.rolling(100).mean()
    rel_ma200 = rel_spx.rolling(200).mean()

    vix = close["^VIX"]
    vix_ratio = vix / vix.rolling(60).mean()
    tnx_chg_60 = close["^TNX"].diff(60)

    # ==========================================
    # v2.0 四维度方向打分（连续平滑，剔除短均线与VIX底部反向噪音）
    # ==========================================
    # 1. 长期趋势与动量 (45%)：波动率归一化均线距离 + 50/200 均线结构 + 6个月动量
    vol_norm = rv60.clip(0.15, 0.55)
    z_ma100 = np.tanh((bench_idx / ma100 - 1) / (vol_norm * np.sqrt(100 / 252) * 0.75))
    z_ma200 = np.tanh((bench_idx / ma200 - 1) / (vol_norm * np.sqrt(200 / 252) * 0.75))
    z_cross = np.tanh((ma50 / ma200 - 1) * 10.0)
    z_mom126 = np.tanh(bench_idx.pct_change(126) / (vol_norm * np.sqrt(126 / 252)))
    df["score_trend"] = (0.20 * z_ma100 + 0.30 * z_ma200 + 0.25 * z_cross + 0.25 * z_mom126).clip(-1, 1)

    # 2. 科技相对标普 500 领导力 (25%)：样本内外均最稳定的正向 alpha 体制因子
    z_rel100 = np.tanh((rel_spx / rel_ma100 - 1) * 12.0)
    z_rel200 = np.tanh((rel_spx / rel_ma200 - 1) * 8.0)
    df["score_rel"] = (0.55 * z_rel100 + 0.45 * z_rel200).clip(-1, 1)

    # 3. 持续实现波动率体制 (15%)：60日波动率区分平静牛市与结构性高波熊市
    df["score_volreg"] = np.tanh((0.28 - rv60) / 0.10).clip(-1, 1)

    # 4. 信用趋势与非对称利率冲击 (15%)：高收益债信用趋势为主，仅对利率急升扣分
    s_credit = (
        0.55 * np.tanh((credit_ratio / cr_ma100 - 1) * 30.0)
        + 0.45 * np.tanh((credit_ratio / cr_ma150 - 1) * 22.0)
    ).clip(-1, 1)
    rate_penalty = np.clip(-(tnx_chg_60 - 0.35).clip(lower=0) / 0.80, -1.0, 0.0)
    df["score_macro"] = (s_credit + 0.35 * rate_penalty).clip(-1, 1)

    # 综合得分 S（12 日 EMA 平滑防高频磨损）
    df["S_raw"] = (
        0.45 * df["score_trend"]
        + 0.25 * df["score_rel"]
        + 0.15 * df["score_volreg"]
        + 0.15 * df["score_macro"]
    ).clip(-1, 1)
    df["S"] = df["S_raw"].ewm(span=CONFIG["ema_span"], adjust=False).mean().clip(-1, 1)

    # ==========================================
    # Net 敞口映射 + 三重结构化门控（Regime Gates）
    # ==========================================
    df["raw_net"] = df["S"].apply(map_net)

    secular_bear = (bench_idx < ma200) & (ma50 < ma200)
    secular_bull = (bench_idx > ma200) & (ma50 > ma200) & (rel_spx > rel_ma200) & (rv60 < 0.35)
    vix_capitulation = vix > CONFIG["vix_capitulation"]

    gated_net = np.where(secular_bear, np.minimum(df["raw_net"], CONFIG["bear_net_cap"]), df["raw_net"])
    gated_net = np.where(secular_bull, np.maximum(gated_net, CONFIG["bull_net_floor"]), gated_net)
    gated_net = np.where(vix_capitulation & (gated_net < 0), 0.0, gated_net)
    df["gated_net"] = gated_net

    df["gate_secular_bear"] = secular_bear
    df["gate_secular_bull"] = secular_bull
    df["gate_vix_capitulation"] = vix_capitulation

    # ==========================================
    # Gross 杠杆风控层（与方向解耦：由 20D 实现波动率 + VIX 短期恐慌驱动）
    # ==========================================
    vol_scalar = (CONFIG["target_vol_gross"] / rv20).clip(0.42, 1.15)
    panic_penalty = np.where((vix_ratio > 1.30) | (vix > 32), 0.80, 1.0)
    base_gross = np.where(df["S"] >= 0.15, 1.50, np.where(df["S"] >= -0.25, 1.15, 0.85))
    df["target_gross"] = (base_gross * vol_scalar * panic_penalty).clip(
        CONFIG["min_gross"], CONFIG["max_gross"]
    )

    df["target_net"] = np.clip(df["gated_net"], -df["target_gross"], df["target_gross"])

    # 15% 调仓缓冲区（No-Trade Buffer）
    exec_net = [0.50]
    for val in df["target_net"].iloc[1:]:
        if pd.isna(val):
            exec_net.append(exec_net[-1])
        elif abs(val - exec_net[-1]) >= CONFIG["rebalance_buffer"]:
            exec_net.append(float(val))
        else:
            exec_net.append(exec_net[-1])
    df["exec_net"] = exec_net
    df["target_long"] = (df["target_gross"] + df["exec_net"]) / 2
    df["target_short"] = (df["target_gross"] - df["exec_net"]) / 2

    # ==========================================
    # 回测引擎（200 日均线预热后，严格 T-1 信号执行 T 日收益）
    # ==========================================
    df["ma200_ready"] = ma200
    bt = df.dropna().copy()
    bt["bench_ret"] = ret_bench.loc[bt.index]
    bt["qqq_ret"] = ret_qqq.loc[bt.index]
    bt["rf"] = rf_daily.loc[bt.index]

    pos = bt["exec_net"].shift(1).fillna(0.5)
    turnover = pos.diff().abs().fillna(0)
    cash_weight = 1.0 - pos
    interest = np.where(
        cash_weight >= 0,
        cash_weight * bt["rf"],
        cash_weight * (bt["rf"] + CONFIG["margin_spread"] / 252.0),
    )
    bt["strat_ret"] = pos * bt["bench_ret"] + interest - turnover * CONFIG["tx_cost"]
    bt["strat_nav"] = (1 + bt["strat_ret"]).cumprod()
    bt["bench_nav"] = (1 + bt["bench_ret"]).cumprod()
    bt["qqq_nav"] = (1 + bt["qqq_ret"]).cumprod()
    bt["strat_dd"] = bt["strat_nav"] / bt["strat_nav"].cummax() - 1
    bt["bench_dd"] = bt["bench_nav"] / bt["bench_nav"].cummax() - 1

    annual_turnover = float(turnover.sum() / (len(bt) / 252.0))
    rebalance_count = int((turnover > 0).sum())

    # 关键周期统计
    crises_def = [
        ("2000-2002 互联网泡沫", "2000-03-01", "2002-10-31"),
        ("2008 全球金融危机", "2007-11-01", "2009-03-09"),
        ("2018 Q4 加息缩表回调", "2018-10-01", "2018-12-24"),
        ("2020 疫情熔断", "2020-02-19", "2020-03-23"),
        ("2022 美联储加息熊市", "2021-12-27", "2022-11-03"),
        ("2025 Q1 关税冲击回调", "2025-02-19", "2025-04-08"),
        ("2023-至今 AI 周期", "2023-01-01", bt.index[-1].strftime("%Y-%m-%d")),
    ]
    crises_out = []
    for name, s_dt, e_dt in crises_def:
        sub = bt.loc[s_dt:e_dt]
        if len(sub) > 5:
            crises_out.append(
                {
                    "name": name,
                    "period": f"{s_dt[:7]} ~ {e_dt[:7]}",
                    "bench_ret": round(float((1 + sub["bench_ret"]).prod() - 1), 4),
                    "strat_ret": round(float((1 + sub["strat_ret"]).prod() - 1), 4),
                    "avg_net": round(float(sub["exec_net"].mean()), 4),
                }
            )

    major_events = extract_major_rebalances(bt, threshold=0.25, cooldown_days=20)

    # 样本外 (2016-至今) 独立校验指标
    bt_oos = bt.loc["2016-01-01":]
    oos_metrics = {
        "strategy": calc_metrics(bt_oos["strat_ret"], bt_oos["rf"]),
        "bench": calc_metrics(bt_oos["bench_ret"], bt_oos["rf"]),
        "qqq": calc_metrics(bt_oos["qqq_ret"], bt_oos["rf"]),
    }

    # 图表抽样（每周最后一个真实交易日，保证无未来日期标签且包含最新交易日）
    weekly = bt.groupby(bt.index.to_period("W-FRI")).tail(1)

    latest_row = bt.iloc[-1]
    latest_date = bt.index[-1].strftime("%Y-%m-%d")

    metrics_obj = {
        "qqq": calc_metrics(bt["qqq_ret"], bt["rf"]),
        "bench": calc_metrics(bt["bench_ret"], bt["rf"]),
        "strategy": calc_metrics(bt["strat_ret"], bt["rf"]),
    }

    payload = {
        "model_version": CONFIG["model_version"],
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "backtest_range": {
            "start": bt.index[0].strftime("%Y-%m-%d"),
            "end": latest_date,
            "trading_days": int(len(bt)),
            "annual_turnover": round(annual_turnover, 2),
            "rebalance_count": rebalance_count,
        },
        "latest": {
            "date": latest_date,
            "qqq_price": round(float(close["QQQ"].iloc[-1]), 2),
            "ndx_price": round(float(close["^NDX"].iloc[-1]), 2),
            "vix": round(float(close["^VIX"].iloc[-1]), 2),
            "us10y": round(float(close["^TNX"].iloc[-1]), 2),
            "realized_vol_20d": round(float(rv20.iloc[-1]), 4),
            "realized_vol_60d": round(float(rv60.iloc[-1]), 4),
            "S": round(float(latest_row["S"]), 3),
            "S_raw": round(float(latest_row["S_raw"]), 3),
            "scores": {
                "trend": round(float(latest_row["score_trend"]), 3),
                "rel_spx": round(float(latest_row["score_rel"]), 3),
                "vol_regime": round(float(latest_row["score_volreg"]), 3),
                "macro": round(float(latest_row["score_macro"]), 3),
            },
            "gates": {
                "secular_bull": bool(latest_row["gate_secular_bull"]),
                "secular_bear": bool(latest_row["gate_secular_bear"]),
                "vix_capitulation": bool(latest_row["gate_vix_capitulation"]),
            },
            "raw_net": round(float(latest_row["raw_net"]), 4),
            "target_net": round(float(latest_row["target_net"]), 4),
            "exec_net": round(float(latest_row["exec_net"]), 4),
            "target_gross": round(float(latest_row["target_gross"]), 4),
            "target_long": round(float(latest_row["target_long"]), 4),
            "target_short": round(float(latest_row["target_short"]), 4),
            "regime": get_regime(float(latest_row["S"])),
        },
        "metrics": metrics_obj,
        "oos_metrics": oos_metrics,
        "crises": crises_out,
        "major_events": major_events,
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "strat_nav": [round(float(x), 3) for x in weekly["strat_nav"]],
            "bench_nav": [round(float(x), 3) for x in weekly["bench_nav"]],
            "qqq_nav": [round(float(x), 3) for x in weekly["qqq_nav"]],
            "exec_net": [round(float(x * 100), 1) for x in weekly["exec_net"]],
            "strat_dd": [round(float(x * 100), 1) for x in weekly["strat_dd"]],
            "bench_dd": [round(float(x * 100), 1) for x in weekly["bench_dd"]],
        },
        "daily_net": {
            "dates": [d.strftime("%Y-%m-%d") for d in bt.index],
            "exec_net": [round(float(x * 100), 1) for x in bt["exec_net"]],
            "target_gross": [round(float(x * 100), 1) for x in bt["target_gross"]],
            "ndx": [round(float(x), 1) for x in close.loc[bt.index, "^NDX"]],
            "S": [round(float(x), 2) for x in bt["S"]],
        },
    }

    os.makedirs("data", exist_ok=True)
    with open("data/latest_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(
        f"\n=== v2.0 全周期回测绩效 ({bt.index[0].strftime('%Y-%m-%d')} ~ {latest_date}, 共 {len(bt)} 交易日) ==="
    )
    for label, key in [
        ("量化敞口策略 v2.0 (仅Beta)", "strategy"),
        ("科技+半导体基准 (70/30)", "bench"),
        ("QQQ (100% 满仓)", "qqq"),
    ]:
        m = metrics_obj[key]
        print(
            f"  {label:24s} | CAGR={m['cagr']:6.2%} | Vol={m['vol']:6.2%} | "
            f"Sharpe={m['sharpe']:5.2f} | MaxDD={m['mdd']:7.2%} | Calmar={m['calmar']:5.2f}"
        )
    print(f"  策略年化换手率: {annual_turnover:.2f}x | 总调仓次数: {rebalance_count}")

    print("\n=== 样本外校验 (2016-01-01 ~ 至今) ===")
    for label, key in [
        ("量化敞口策略 v2.0 (OOS)", "strategy"),
        ("科技+半导体基准 (OOS)", "bench"),
        ("QQQ 满仓 (OOS)", "qqq"),
    ]:
        m = oos_metrics[key]
        print(
            f"  {label:24s} | CAGR={m['cagr']:6.2%} | Vol={m['vol']:6.2%} | "
            f"Sharpe={m['sharpe']:5.2f} | MaxDD={m['mdd']:7.2%} | Calmar={m['calmar']:5.2f}"
        )

    print("\n=== 关键历史周期对比 ===")
    for c in crises_out:
        print(
            f"  {c['name']:20s} | 基准: {c['bench_ret']:+7.1%} | 策略: {c['strat_ret']:+7.1%} | 平均Net: {c['avg_net']:+6.1%}"
        )

    print(f"\n=== 重大加减仓事件日志 (共 {len(major_events)} 次, 最近 15 次) ===")
    for ev in major_events[-15:]:
        print(
            f"  {ev['date']} | {ev['action']} | Net: {ev['net_before']:+5.0%} -> {ev['net_after']:+5.0%} "
            f"({ev['delta_net']:+5.0%}) | Gross: {ev['gross_after']:4.0%} | S={ev['score_S']:+.2f} | 主导: {ev['main_driver']}"
        )

    l = payload["latest"]
    print("\n=== 最新交易日仓位指令 ===")
    print(f"  日期: {l['date']} | 状态: {l['regime']['label']} | S={l['S']:+.3f} (未平滑: {l['S_raw']:+.3f})")
    print(
        f"  分项得分 -> 长期趋势: {l['scores']['trend']:+.3f} | 相对标普: {l['scores']['rel_spx']:+.3f} | "
        f"波动体制: {l['scores']['vol_regime']:+.3f} | 信用利率: {l['scores']['macro']:+.3f}"
    )
    print(
        f"  结构门控 -> 牛市底仓: {l['gates']['secular_bull']} | 熊市封顶: {l['gates']['secular_bear']} | VIX>40平空: {l['gates']['vix_capitulation']}"
    )
    print(
        f"  Target Net (执行): {l['exec_net']:+.1%} (原始目标: {l['target_net']:+.1%}) | "
        f"Target Gross: {l['target_gross']:.1%} | 参考配比: L {l['target_long']:.1%} / S {l['target_short']:.1%}"
    )
    print(
        f"  行情快照 -> QQQ: ${l['qqq_price']:.2f} | NDX: {l['ndx_price']:.2f} | "
        f"VIX: {l['vix']:.2f} | US10Y: {l['us10y']:.2f}% | 20D/60D Vol: {l['realized_vol_20d']:.1%} / {l['realized_vol_60d']:.1%}"
    )
    print(f"\nSuccessfully generated data/latest_data.json for {latest_date} (model {CONFIG['model_version']})")

    send_notifications(payload)


if __name__ == "__main__":
    main()
