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
    "model_version": "v3.1",
    "start_date": "1999-03-10",
    "max_gross": 1.50,          # Gross 上限 150%
    "min_gross": 0.00,          # 灵活 Gross 下限 0%（允许极端混乱期全现金观望）
    "cash_mode_rv20": 0.45,     # 全现金触发线：共识转弱 (S<0) 且 20D 波动率 > 45%
    "max_net": 1.20,            # 强共识最高 Net +120%
    "min_net": -0.25,           # 共识瓦解最低 Net -25%
    "neutral_anchor_net": 0.45, # Long-Biased 偏多中枢：S=0 对应 +45% Net
    "score_full_scale": 0.65,   # S=±0.65 触及上下限，避免双曲正切末端钝化
    "target_vol_gross": 0.28,   # 独立 Gross 目标波动率 28%（仅由 RV20 驱动）
    "ema_span": 10,             # 综合共识得分 10 日 EMA 平滑
    "rebalance_buffer": 0.15,   # 15% Net 调仓缓冲区（年化换手约 1.8x）
    "margin_spread": 0.015,     # 杠杆融资点差（无风险利率 + 1.5%）
    "tx_cost": 0.0005,          # 单边交易滑点与冲击成本 5bp
}

CORE_TICKERS = ["QQQ", "^SOX", "^NDX", "^GSPC", "VWEHX", "VFITX", "^IRX", "^VIX"]
AUX_TICKERS = ["^VIX3M"]  # 2006-07 起有数据，仅作实盘辅助雷达，不截断 1999 回测


def map_net(s: float) -> float:
    """单一平滑非对称映射（零硬门控）：S=0 -> +45%，S=+0.65 -> +120%，S=-0.65 -> -25%"""
    anchor = CONFIG["neutral_anchor_net"]
    scale = CONFIG["score_full_scale"]
    if s >= 0.0:
        return anchor + min(1.0, s / scale) * (CONFIG["max_net"] - anchor)
    else:
        return max(CONFIG["min_net"], anchor + (s / scale) * (anchor - CONFIG["min_net"]))


def get_regime(s: float, cash_mode: bool = False) -> dict:
    if cash_mode:
        return {
            "label": "全现金观望（极端高波转弱）",
            "color": "#c084fc",
            "action": "共识转弱 (S<0) 叠加 20 日波动率 >45%：清仓至 Gross 0% / Net 0% 持币吃短债息，彻底避开多空双杀",
        }
    if s >= 0.35:
        return {
            "label": "强多头（主升浪共识）",
            "color": "#10b981",
            "action": "资金与趋势高度共振，高净多进攻，容忍常规洗盘",
        }
    elif s >= 0.00:
        return {
            "label": "温和偏多 / 震荡蓄势",
            "color": "#38bdf8",
            "action": "保持显著偏多配置，精选 AI/科技核心个股，减少折腾",
        }
    elif s >= -0.42:
        return {
            "label": "中性防御 / 共识转弱",
            "color": "#f59e0b",
            "action": "用 QQQ/MNQ 对冲多余 Beta，聚焦纯 Alpha 并关注下方辅助雷达",
        }
    else:
        return {
            "label": "确认熊市 / 共识瓦解",
            "color": "#ef4444",
            "action": "允许适度净空保护净值并严控 Gross；若 VIX/VIX3M 倒挂则勿追空",
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
                "科技长期趋势": 0.65 * row["score_trend"],
                "科技相对标普": 0.35 * row["score_lead"],
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


def download_close_data(core_tickers: list, aux_tickers: list, start_date: str) -> pd.DataFrame:
    all_tickers = core_tickers + aux_tickers
    raw = yf.download(
        all_tickers,
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

    missing = [t for t in all_tickers if t not in close.columns or close[t].dropna().empty]
    for t in missing:
        single = yf.download(t, start=start_date, auto_adjust=True, progress=False)
        if single is not None and not single.empty:
            s_close = single["Close"]
            if isinstance(s_close, pd.DataFrame):
                s_close = s_close.iloc[:, 0]
            close[t] = s_close

    still_missing = [t for t in core_tickers if t not in close.columns or close[t].dropna().empty]
    if still_missing:
        raise RuntimeError(f"以下核心标的数据下载失败: {still_missing}")

    close = close.sort_index()
    print("=== 各标的数据覆盖范围 ===")
    for t in all_tickers:
        if t in close.columns and not close[t].dropna().empty:
            s = close[t].dropna()
            print(f"  {t:7s}: {s.index[0].strftime('%Y-%m-%d')} ~ {s.index[-1].strftime('%Y-%m-%d')} ({len(s)} rows)")
        else:
            print(f"  {t:7s}: 缺失（仅辅助项）")

    # 严格以 QQQ 真实美股交易日历为基准，且不让 2006 年才起步的 ^VIX3M 截断 1999 回测
    qqq_dates = close["QQQ"].dropna().index
    core_df = close[core_tickers].ffill().loc[qqq_dates].dropna()
    for t in aux_tickers:
        if t in close.columns:
            core_df[t] = close[t].ffill().reindex(core_df.index)
        else:
            core_df[t] = np.nan
    return core_df


def build_radar_snapshot(close: pd.DataFrame, bench_idx: pd.Series) -> dict:
    """构建独立辅助决策雷达（不进入主公式，用于人工极端情况辅助加减仓）"""
    # 1. 信用利差代理：VWEHX (高收益债) / VFITX (中期美债) 偏离 120D 均线 (%)
    cr = close["VWEHX"] / close["VFITX"]
    cr_ma120 = cr.rolling(120).mean()
    cr_dev_series = (cr / cr_ma120 - 1.0).dropna()
    cr_cur = float(cr.iloc[-1])
    cr_ma_cur = float(cr_ma120.iloc[-1])
    cr_dev = float(cr_dev_series.iloc[-1])
    cr_chg20 = float(cr.pct_change(20).iloc[-1])
    cr_mean = float(cr_dev_series.mean())
    cr_std = float(cr_dev_series.std())
    cr_z = (cr_dev - cr_mean) / cr_std if cr_std > 0 else 0.0

    if cr_dev <= -0.015 or (cr_dev < 0 and cr_chg20 <= -0.01):
        cr_level, cr_label, cr_color = "danger", "红灯 · 信用显著收紧（系统性抽水预警）", "#ef4444"
        cr_hint = "信用市场已明显走弱，若主模型 Net 仍偏高，可考虑提前手动多降 15%~25% Net 并压低 Gross，严防 2008 式信贷收缩。"
    elif cr_dev < 0:
        cr_level, cr_label, cr_color = "warn", "黄灯 · 信用温和走弱（留意机构去风险）", "#f59e0b"
        cr_hint = "信用比率小幅低于 120 日均线，暂无需激进干预，若继续下行跌破 -1.5% 再考虑额外降仓。"
    else:
        cr_level, cr_label, cr_color = "ok", "绿灯 · 信用环境健康（Risk-On）", "#10b981"
        cr_hint = "高收益债相对国债位于 120 日均线上方，金融流动性正常；若刚经历大熊市后率先转绿，可增强右侧加仓信心。"

    # 2. 波动率期限结构：^VIX / ^VIX3M
    vix = close["^VIX"]
    vix3m = close["^VIX3M"]
    vix_term = vix / vix3m
    vt_valid = vix_term.dropna()
    vix_cur = float(vix.iloc[-1])
    vix3m_cur = float(vix3m.iloc[-1]) if pd.notna(vix3m.iloc[-1]) else None
    vt_cur = float(vix_term.iloc[-1]) if pd.notna(vix_term.iloc[-1]) else None
    vt_mean = float(vt_valid.mean()) if not vt_valid.empty else None
    vt_std = float(vt_valid.std()) if not vt_valid.empty else None
    vt_z = ((vt_cur - vt_mean) / vt_std) if (vt_cur is not None and vt_mean is not None and vt_std and vt_std > 0) else None

    ma50 = bench_idx.rolling(50).mean()
    above_ma50 = bool(bench_idx.iloc[-1] > ma50.iloc[-1])
    inv_series = vix_term > 1.03
    inv_recent_40d = bool(inv_series.iloc[-40:].any()) if vix_term.dropna().shape[0] >= 5 else False
    max_vt_40d = float(vix_term.iloc[-40:].max()) if vix_term.iloc[-40:].notna().any() else None

    if vt_cur is None:
        vt_level, vt_label, vt_color = "warn", "数据暂缺（^VIX3M 未返回）", "#94a3b8"
        vt_hint = "暂未获取到 ^VIX3M 数据，请以主模型信号与 VIX 绝对水位为准。"
    elif vt_cur >= 1.03:
        vt_level, vt_label, vt_color = "danger", "红灯 · 近月恐慌倒挂（严禁追空 / 控 Gross）", "#ef4444"
        vt_hint = "VIX/VIX3M ≥ 1.03 代表短期恐慌倒挂。实证显示倒挂后 60 日科技股平均反弹 +6.8%：此时应压低 Gross 防多空双杀，但切忌追空或新建净空头！"
    elif inv_recent_40d and vt_cur < 0.94 and above_ma50:
        vt_level, vt_label, vt_color = "ok", "绿灯 · 恐慌解除右侧加仓确认", "#10b981"
        vt_hint = "近 40 日出现过倒挂，现已回落至 <0.94 且价格站回 50 日均线！说明恐慌对冲盘出清，若当前 Net 低于 +65%，可考虑手动提前分批回补多头。"
    elif vt_cur >= 0.96:
        vt_level, vt_label, vt_color = "warn", "黄灯 · 短期避险升温（保持观察）", "#f59e0b"
        vt_hint = "期限结构偏平（0.96~1.03），市场情绪偏紧；按主模型指令执行即可，等待倒挂或明确回落信号。"
    else:
        vt_level, vt_label, vt_color = "normal", "正常 · 期限结构贴水（Contango 平静期）", "#38bdf8"
        vt_hint = "VIX 处于正常 Contango 状态（<0.96），无恐慌事件干扰，严格跟随主模型 2 因子信号即可。"

    return {
        "credit": {
            "ratio": round(cr_cur, 4),
            "ma120": round(cr_ma_cur, 4),
            "dev_pct": round(cr_dev, 4),
            "chg_20d_pct": round(cr_chg20, 4),
            "hist_mean_pct": round(cr_mean, 4),
            "hist_std_pct": round(cr_std, 4),
            "band_upper_pct": round(cr_mean + cr_std, 4),
            "band_lower_pct": round(cr_mean - cr_std, 4),
            "z_score": round(cr_z, 2),
            "level": cr_level,
            "label": cr_label,
            "color": cr_color,
            "hint": cr_hint,
        },
        "vix_term": {
            "vix": round(vix_cur, 2),
            "vix3m": round(vix3m_cur, 2) if vix3m_cur is not None else None,
            "ratio": round(vt_cur, 3) if vt_cur is not None else None,
            "max_40d": round(max_vt_40d, 3) if max_vt_40d is not None else None,
            "hist_mean": round(vt_mean, 3) if vt_mean is not None else None,
            "hist_std": round(vt_std, 3) if vt_std is not None else None,
            "band_upper": round(vt_mean + vt_std, 3) if (vt_mean is not None and vt_std is not None) else None,
            "band_lower": round(vt_mean - vt_std, 3) if (vt_mean is not None and vt_std is not None) else None,
            "z_score": round(vt_z, 2) if vt_z is not None else None,
            "inverted_recent_40d": inv_recent_40d,
            "above_ma50": above_ma50,
            "level": vt_level,
            "label": vt_label,
            "color": vt_color,
            "hint": vt_hint,
        },
    }


def send_notifications(payload: dict):
    latest = payload["latest"]
    radar = payload["radar"]
    vt = radar["vix_term"]
    cr = radar["credit"]
    vt_str = f"{vt['ratio']:.3f}" if vt["ratio"] is not None else "--"

    msg = (
        f"📊 科技/AI 量化敞口日报 {CONFIG['model_version']} 极简共识版 ({latest['date']})\n"
        f"状态：{latest['regime']['label']}\n"
        f"综合共识得分 S：{latest['S']:+.2f}\n"
        f"• 科技长期趋势 (65%): {latest['scores']['trend']:+.2f}\n"
        f"• 科技相对标普 (35%): {latest['scores']['lead']:+.2f}\n"
        f"——————————————\n"
        f"🎯 Target Net：{latest['exec_net']:+.1%}（原始 {latest['target_net']:+.1%}）\n"
        f"⚖️ Target Gross：{latest['target_gross']:.1%}（20D 波动 {latest['realized_vol_20d']:.1%}）\n"
        f"📌 参考配比：Long {latest['target_long']:.1%} / Short {latest['target_short']:.1%}\n"
        f"——————————————\n"
        f"🛰 辅助雷达（参考）：\n"
        f"• 信用利差偏离120D: {cr['dev_pct']:+.2%} ({cr['label']})\n"
        f"• VIX/VIX3M: {vt_str} ({vt['label']})"
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
    close = download_close_data(CORE_TICKERS, AUX_TICKERS, CONFIG["start_date"])

    ret_qqq = close["QQQ"].pct_change().fillna(0)
    ret_sox = close["^SOX"].pct_change().fillna(0)
    ret_bench = 0.70 * ret_qqq + 0.30 * ret_sox
    bench_idx = (1 + ret_bench).cumprod()
    rf_daily = (close["^IRX"] / 100.0).clip(lower=0) / 252.0
    spx = close["^GSPC"]

    df = pd.DataFrame(index=close.index)

    # 基础序列（严格无未来函数）
    ma50 = bench_idx.rolling(50).mean()
    ma200 = bench_idx.rolling(200).mean()
    rv20 = ret_bench.rolling(20).std() * np.sqrt(252)
    rv60 = ret_bench.rolling(60).std() * np.sqrt(252)
    rel_spx = bench_idx / spx
    rel_ma150 = rel_spx.rolling(150).mean()

    # ==========================================
    # v3.0 极简 2 因子市场共识打分（零门控、零冗余补丁）
    # ==========================================
    # 1. 绝对共识：科技长期趋势 (65%) = 200D 波动率归一化均线距离 + 50/200 结构
    vol_norm = rv60.clip(0.15, 0.55)
    z_ma200 = np.tanh((bench_idx / ma200 - 1) / (vol_norm * np.sqrt(200 / 252) * 0.75))
    z_cross = np.tanh((ma50 / ma200 - 1) * 10.0)
    df["score_trend"] = (0.60 * z_ma200 + 0.40 * z_cross).clip(-1, 1)

    # 2. 相对共识：科技相对标普 500 领导力 (35%)
    df["score_lead"] = np.tanh((rel_spx / rel_ma150 - 1) * 10.0).clip(-1, 1)

    # 综合共识得分 S（10 日 EMA 平滑）
    df["S_raw"] = (0.65 * df["score_trend"] + 0.35 * df["score_lead"]).clip(-1, 1)
    df["S"] = df["S_raw"].ewm(span=CONFIG["ema_span"], adjust=False).mean().clip(-1, 1)

    # ==========================================
    # 单一平滑 Net 映射 + 独立波动率 Gross 风控层 (v3.1 方案 D: 0%~150% + 全现金模式)
    # ==========================================
    df["raw_net"] = df["S"].apply(map_net)

    # Gross 仅由 20D 实现波动率目标法驱动（下限放开至 0%）
    vol_scalar = (CONFIG["target_vol_gross"] / rv20).clip(0.00, 1.00)
    base_gross = (CONFIG["max_gross"] * vol_scalar).clip(CONFIG["min_gross"], CONFIG["max_gross"])

    # 极端高波转弱全现金观望模式：当共识转弱 (S < 0) 且 20D 实现波动率 > 45% -> 强制 Gross=0%, Net=0%
    cash_mode_series = (df["S"] < 0.0) & (rv20 > CONFIG["cash_mode_rv20"])
    df["cash_mode"] = cash_mode_series
    df["target_gross"] = np.where(cash_mode_series, 0.0, base_gross)
    raw_net_eff = np.where(cash_mode_series, 0.0, df["raw_net"])
    df["target_net"] = np.clip(raw_net_eff, -df["target_gross"], df["target_gross"])

    # 15% 调仓缓冲区（全现金模式下强制归零，确保 |exec_net| <= target_gross=0）
    exec_net = [0.50]
    for i in range(1, len(df)):
        val = df["target_net"].iloc[i]
        cm = bool(cash_mode_series.iloc[i])
        if pd.isna(val):
            exec_net.append(exec_net[-1])
        elif cm:
            exec_net.append(0.0)
        elif abs(val - exec_net[-1]) >= CONFIG["rebalance_buffer"]:
            exec_net.append(float(val))
        else:
            exec_net.append(exec_net[-1])
    df["exec_net"] = exec_net
    # 确保逻辑上参考多空始终满足 Gross >= |exec_net|（若缓冲保留了旧 Net 且未触发全现金，显示上界）
    eff_gross_disp = np.where(cash_mode_series, 0.0, np.maximum(df["target_gross"], np.abs(df["exec_net"])))
    df["target_long"] = (eff_gross_disp + df["exec_net"]) / 2
    df["target_short"] = (eff_gross_disp - df["exec_net"]) / 2

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
    net_short_pct = float((bt["exec_net"] < 0).mean())
    cash_mode_pct = float(bt["cash_mode"].mean())

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
    radar_snapshot = build_radar_snapshot(close, bench_idx)

    bt_oos = bt.loc["2016-01-01":]
    oos_metrics = {
        "strategy": calc_metrics(bt_oos["strat_ret"], bt_oos["rf"]),
        "bench": calc_metrics(bt_oos["bench_ret"], bt_oos["rf"]),
        "qqq": calc_metrics(bt_oos["qqq_ret"], bt_oos["rf"]),
    }

    weekly = bt.groupby(bt.index.to_period("W-FRI")).tail(1)
    latest_row = bt.iloc[-1]
    latest_date = bt.index[-1].strftime("%Y-%m-%d")

    metrics_obj = {
        "qqq": calc_metrics(bt["qqq_ret"], bt["rf"]),
        "bench": calc_metrics(bt["bench_ret"], bt["rf"]),
        "strategy": calc_metrics(bt["strat_ret"], bt["rf"]),
    }

    # 供辅助雷达与核心 2 因子历史趋势图使用的对齐日频序列与统计
    cr_ratio_full = close["VWEHX"] / close["VFITX"]
    cr_dev_full = (cr_ratio_full / cr_ratio_full.rolling(120).mean() - 1.0) * 100.0
    vix_term_full = close["^VIX"] / close["^VIX3M"]

    tr_mean = float(bt["score_trend"].mean())
    tr_std = float(bt["score_trend"].std())
    tr_z = (float(latest_row["score_trend"]) - tr_mean) / tr_std if tr_std > 0 else 0.0

    ld_mean = float(bt["score_lead"].mean())
    ld_std = float(bt["score_lead"].std())
    ld_z = (float(latest_row["score_lead"]) - ld_mean) / ld_std if ld_std > 0 else 0.0

    payload = {
        "model_version": CONFIG["model_version"],
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "backtest_range": {
            "start": bt.index[0].strftime("%Y-%m-%d"),
            "end": latest_date,
            "trading_days": int(len(bt)),
            "annual_turnover": round(annual_turnover, 2),
            "rebalance_count": rebalance_count,
            "net_short_pct": round(net_short_pct, 4),
            "cash_mode_pct": round(cash_mode_pct, 4),
        },
        "latest": {
            "date": latest_date,
            "qqq_price": round(float(close["QQQ"].iloc[-1]), 2),
            "ndx_price": round(float(close["^NDX"].iloc[-1]), 2),
            "vix": round(float(close["^VIX"].iloc[-1]), 2),
            "realized_vol_20d": round(float(rv20.iloc[-1]), 4),
            "realized_vol_60d": round(float(rv60.iloc[-1]), 4),
            "S": round(float(latest_row["S"]), 3),
            "S_raw": round(float(latest_row["S_raw"]), 3),
            "scores": {
                "trend": round(float(latest_row["score_trend"]), 3),
                "lead": round(float(latest_row["score_lead"]), 3),
            },
            "factor_details": {
                "bench_vs_ma200_pct": round(float(bench_idx.iloc[-1] / ma200.iloc[-1] - 1.0), 4),
                "ma50_vs_ma200_pct": round(float(ma50.iloc[-1] / ma200.iloc[-1] - 1.0), 4),
                "rel_vs_ma150_pct": round(float(rel_spx.iloc[-1] / rel_ma150.iloc[-1] - 1.0), 4),
            },
            "factor_stats": {
                "trend": {
                    "hist_mean": round(tr_mean, 3),
                    "hist_std": round(tr_std, 3),
                    "band_upper": round(min(1.0, tr_mean + tr_std), 3),
                    "band_lower": round(max(-1.0, tr_mean - tr_std), 3),
                    "z_score": round(tr_z, 2),
                },
                "lead": {
                    "hist_mean": round(ld_mean, 3),
                    "hist_std": round(ld_std, 3),
                    "band_upper": round(min(1.0, ld_mean + ld_std), 3),
                    "band_lower": round(max(-1.0, ld_mean - ld_std), 3),
                    "z_score": round(ld_z, 2),
                },
            },
            "raw_net": round(float(latest_row["raw_net"]), 4),
            "target_net": round(float(latest_row["target_net"]), 4),
            "exec_net": round(float(latest_row["exec_net"]), 4),
            "target_gross": round(float(latest_row["target_gross"]), 4),
            "target_long": round(float(latest_row["target_long"]), 4),
            "target_short": round(float(latest_row["target_short"]), 4),
            "cash_mode": bool(latest_row["cash_mode"]),
            "regime": get_regime(float(latest_row["S"]), bool(latest_row["cash_mode"])),
        },
        "radar": radar_snapshot,
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
            "strat_nav": [round(float(x), 3) for x in bt["strat_nav"]],
            "ndx": [round(float(x), 1) for x in close.loc[bt.index, "^NDX"]],
            "S": [round(float(x), 2) for x in bt["S"]],
            "score_trend": [round(float(x), 3) for x in bt["score_trend"]],
            "score_lead": [round(float(x), 3) for x in bt["score_lead"]],
            "credit_dev": [
                round(float(x), 2) if pd.notna(x) else None for x in cr_dev_full.loc[bt.index]
            ],
            "vix_term": [
                round(float(x), 3) if pd.notna(x) else None for x in vix_term_full.loc[bt.index]
            ],
        },
    }

    os.makedirs("data", exist_ok=True)
    with open("data/latest_data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    print(
        f"\n=== {CONFIG['model_version']} 极简共识版 全周期绩效 ({bt.index[0].strftime('%Y-%m-%d')} ~ {latest_date}, 共 {len(bt)} 交易日) ==="
    )
    for label, key in [
        (f"量化敞口策略 {CONFIG['model_version']} (仅Beta)", "strategy"),
        ("科技+半导体基准 (70/30)", "bench"),
        ("QQQ (100% 满仓)", "qqq"),
    ]:
        m = metrics_obj[key]
        print(
            f"  {label:24s} | CAGR={m['cagr']:6.2%} | Vol={m['vol']:6.2%} | "
            f"Sharpe={m['sharpe']:5.2f} | MaxDD={m['mdd']:7.2%} | Calmar={m['calmar']:5.2f}"
        )
    print(f"  策略年化换手率: {annual_turnover:.2f}x | 总调仓次数: {rebalance_count} | Net<0 占比: {net_short_pct:.1%} | 全现金占比: {cash_mode_pct:.1%}")

    print("\n=== 样本外校验 (2016-01-01 ~ 至今) ===")
    for label, key in [
        (f"量化敞口策略 {CONFIG['model_version']} (OOS)", "strategy"),
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

    print(f"\n=== 重大加减仓事件日志 (共 {len(major_events)} 次, 最近 12 次) ===")
    for ev in major_events[-12:]:
        print(
            f"  {ev['date']} | {ev['action']} | Net: {ev['net_before']:+5.0%} -> {ev['net_after']:+5.0%} "
            f"({ev['delta_net']:+5.0%}) | Gross: {ev['gross_after']:4.0%} | S={ev['score_S']:+.2f} | 主导: {ev['main_driver']}"
        )

    l = payload["latest"]
    cr_r = radar_snapshot["credit"]
    vt_r = radar_snapshot["vix_term"]
    print("\n=== 最新交易日仓位指令 ===")
    print(f"  日期: {l['date']} | 状态: {l['regime']['label']} | S={l['S']:+.3f} (未平滑: {l['S_raw']:+.3f})")
    print(f"  共识拆解 -> 长期趋势(65%): {l['scores']['trend']:+.3f} | 相对标普(35%): {l['scores']['lead']:+.3f}")
    print(
        f"  Target Net (执行): {l['exec_net']:+.1%} (原始目标: {l['target_net']:+.1%}) | "
        f"Target Gross: {l['target_gross']:.1%} | 参考配比: L {l['target_long']:.1%} / S {l['target_short']:.1%}"
    )
    print(
        f"  辅助雷达 -> 信用偏离120D: {cr_r['dev_pct']:+.2%} [{cr_r['label']}] | "
        f"VIX/VIX3M: {vt_r['ratio']} [{vt_r['label']}]"
    )
    print(f"\nSuccessfully generated data/latest_data.json for {latest_date} (model {CONFIG['model_version']})")

    send_notifications(payload)


if __name__ == "__main__":
    main()
