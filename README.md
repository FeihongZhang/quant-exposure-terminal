# Tech / AI 量化敞口管理终端（v2.0）

面向美股为主、聚焦 AI/科技股的 **Long-Biased 多空组合**，将“个股 Alpha 研究”与“市场 Beta / 杠杆敞口管理”解耦的系统化敞口管理层（Systematic Exposure Overlay）。

## 核心设计（v2.0）
- **敞口约束**：`Gross 60% ~ 150%` ｜ `Net -25% ~ +120%` ｜ `12 日 EMA 平滑` ｜ `15% Net 调仓缓冲区`
- **四维度方向打分（$S \in [-1, +1]$）**：
  1. **长期趋势与动量（45%）**：`QQQ (70%) + ^SOX (30%)` 波动率归一化 100D/200D 均线距离 + 50/200 结构 + 126D 动量
  2. **科技相对标普领导力（25%）**：`Tech / ^GSPC` 相对 100D/200D 均线强弱
  3. **持续波动率体制（15%）**：60 日实现波动率 `RV60`（区分平静牛市与结构性高波熊市）
  4. **信用趋势与非对称利率冲击（15%）**：`VWEHX / VFITX` 长期趋势 + 美债 10Y 急升单边惩罚
- **三重结构化门控**：
  - 长期熊市防诱多封顶（`P < MA200` 且 `MA50 < MA200` → Net ≤ `+5%`）
  - 长期牛市底仓保护（`P > MA200`、`MA50 > MA200`、科技跑赢标普且 `RV60 < 35%` → Net ≥ `+85%`）
  - 极端恐慌平空保护（`VIX > 40` 不留净空头防逼空）
- **Gross 杠杆风控层（与方向解耦）**：由 20 日实现波动率（目标 28%）与短期 VIX 恐慌惩罚（`VIX > 32` 或 `VIX / MA60 > 1.30` 打 8 折）驱动。

## GitHub 每日自动更新与 Pages 上线步骤
1. 在仓库 **Settings → Actions → General → Workflow permissions** 勾选 **Read and write permissions** 并保存。
2. 在 **Settings → Pages** 中将 Branch 设为 `main` / `/ (root)` 并保存，等待 1–2 分钟生成专属监控网址。
3. （可选）在 **Settings → Secrets and variables → Actions** 添加推送密钥：
   - 飞书：`FEISHU_WEBHOOK_URL`
   - Telegram：`TELEGRAM_BOT_TOKEN`、`TELEGRAM_CHAT_ID`
4. 每个美股交易日收盘前 30 分钟（UTC 19:30 / 20:30），GitHub Actions 会自动运行 `update_data.py` 更新 `data/latest_data.json` 并推送日报。
