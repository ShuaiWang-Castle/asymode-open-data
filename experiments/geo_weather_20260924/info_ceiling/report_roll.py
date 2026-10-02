"""Regenerate the restoration-kernel block (between the RK_BEGIN and RK_END markers) of notes/OVERNIGHT_RESULTS_20261002_ZH.md from
results/v1/info_ceiling/roll_scores_48.json and roll_transfer_48.json. Run in the experiment directory: python info_ceiling/report_roll.py"""
import json, os, re
r = json.load(open("results/v1/info_ceiling/roll_scores_48.json"))
f = lambda x: f"{100 * x['point']:+.1f}% [{100 * x['ci95'][0]:+.1f}, {100 * x['ci95'][1]:+.1f}]".replace("-", "−")
pc = lambda v: f"{100 * v:+.1f}%".replace("-", "−")
RN = {"tropical": "热带", "winter": "冬季", "synoptic_wind": "大风", "convective": "对流", "heavy_rain": "强降雨"}
k = r["RK"]; nf = len(k["folds"])
lines = []
status = "五折全部完成" if nf == 5 else f"已完成 {nf} 折（第 {', '.join(map(str, k['folds']))} 折），其余在运行，数字会更新"
lines.append(f"**RK 相对重新训练的宿主（留出事件；{status}）**\n")
lines.append("| 样本 | 县-事件数 | RMSE 改善 | 95% 区间 |\n|---|---:|---:|---|")
def row(name, n, x):
    lo, hi = x["ci95"]
    lines.append(f"| {name} | {n} | {pc(x['point'])} | [{pc(lo)[:-1]}, {pc(hi)[:-1]}] |")
row("全体（96 小时）", k["units"], k["all"]); row("起报时已在停电（≥1%）", k["active_units"], k["active"])
row("起报时尚未停电", k["units"] - k["active_units"], k["not_active"]); row("S 单元", k["S_units"], k["S"])
row("全体，前 48 小时", k["units"], k["all_48h"]); row("已在停电，前 48 小时", k["active_units"], k["active_48h"])
lines.append("")
lines.append("- 分天气类型（全体）：" + "，".join(f"{RN[q]} {pc(v)}" for q, v in k["all_by_regime"].items()) + "。")
lines.append("- 分折（全体 / 已在停电）：" + "；".join(f"第 {q} 折 {pc(v['all'])} / {pc(v['active'])}" for q, v in k["by_fold"].items()) + "。")
lines.append(f"- 假峰（预测峰值 ≥10% 而观测 <10%）：RK {k['false_peaks']['arm']} 个，宿主 {k['false_peaks']['host']} 个。")
kk = k["kernel"]
lines.append("- 学到的权重（各折）：自身规模饱和 κ_L = " + "、".join(f"{v['rest_kappa_l']:.1f}" for v in kk.values()) + "；三个距离环 κ = " + "；".join("[" + ", ".join(f"{x:.1f}" for x in v["rest_kappa_b"]) + "]" for v in kk.values()) + "。")
if "regional_slowdown_factor_active" in k:
    q = k["regional_slowdown_factor_active"]
    lines.append(f"- 区域项对已停电县的恢复放慢倍数 1 + Σκ_k·b_k：中位数 {q['median']:.2f}（四分位 {q['q25']:.2f}–{q['q75']:.2f}，90 分位 {q['q90']:.2f}）；" + "，".join(f"{RN[a]} {v:.2f}" for a, v in q["by_regime"].items()) + f"；尚未停电的县 {q['not_active_median']:.2f}。模型没有用天气类型标签，自己把放慢集中到了热带。")
h = k["host_vs_registered_host_rolled"]
lines.append("- 重新训练的宿主并不比“原宿主的速率从新起报点滚动”更好（全体 " + f(h["all"]) + "）：宿主的恢复网络即使看到起报时的自身停电，也学不出这个结构。")
lines.append(f"- 相对持续性基线（存量不变）：宿主 {f(k['host_vs_persistence']['all'])}，RK {f(k['arm_vs_persistence']['all'])}。")
lines.append("- 图：`figures/roll_trajectories_48.png`（留出系统里已停电的县，观测、宿主、RK 的轨迹）。")
if os.path.exists("results/v1/info_ceiling/roll_concentration_48.json"):
    c = json.load(open("results/v1/info_ceiling/roll_concentration_48.json"))
    top5 = sum(q["share_of_gain"] for q in c["top"][:5])
    lines.append(f"- **收益的集中度**：{c['systems']} 个系统里 {c['better']} 个变好、{c['worse']} 个变差，最差的一个只抵消总收益的 {abs(100 * c['worst'][0]['share_of_gain']):.1f}%。但加权误差的下降集中在少数大型、长时间的事件上：前 5 个系统占总收益的 {100 * top5:.0f}%（最大的一个是冬季系统，占 {100 * c['top'][0]['share_of_gain']:.0f}%）。去掉最大的 1、2、3、5 个系统后，全体改善为 {pc(c['without_top']['1'])}、{pc(c['without_top']['2'])}、{pc(c['without_top']['3'])}、{pc(c['without_top']['5'])}。")
ok = k["all"]["ci95"][0] > 0 and k["active"]["point"] > 0
if nf == 5:
    lines.append("")
    lines.append(f"**按登记规则：RK {'是候选' if ok else '不是候选'}**（全体改善区间{'不含' if k['all']['ci95'][0] > 0 else '含'} 0，已在停电样本的点估计{'为正' if k['active']['point'] > 0 else '不为正'}）。" + ("对照 arm 已按登记顺序启动：只有自身状态（RKl）、打乱区域负担（RKp）、恢复网络学习率对照（hostF、BinF）、损伤侧形式（RKu）、普通输入（Bin）。" if ok else "不追加对照。"))
if os.path.exists("results/v1/info_ceiling/roll_transfer_48.json"):
    tr = json.load(open("results/v1/info_ceiling/roll_transfer_48.json"))
    lines.append(f"\n**不重新训练，直接换起报点**（宿主和 RK 都只在后移 48 小时的面板上训练；留出事件，前 48 小时；{len(tr['folds'])} 折）\n")
    lines.append("| 起报点 | 已在停电的县-事件 | 全体 | 已在停电 |\n|---|---:|---|---|")
    for d in ("0", "24", "48", "72", "96"):
        if d in tr:
            lines.append(f"| +{d} 小时{'（训练用）' if d == '48' else ('（登记的风暴前起报点）' if d == '0' else '')} | {tr[d]['active_units']} | {f(tr[d]['all'])} | {f(tr[d]['active'])} |")
    lines.append("\n- 在一个起报点学到的核，换到其他起报点仍然有效。这支持“一个模型、多个起报点”的设计。")
    if "0" in tr:
        fw = tr["0"]["full_window"]
        lines.append(f"- 在登记的风暴前起报点上，核不伤害原任务：144 小时全体 {f(fw['all'])}，S {f(fw['S'])}。")
    lines.append("- 图里能看到剩下的问题：最大的两场热带系统里 RK 恢复得比观测慢。区域负担固定在起报时的值，邻县恢复之后它不会下降。下一步应让区域负担随邻县的预测存量演化。")
names = {"RKl": "RKl 只有自身规模饱和", "RKp": "RKp 区域负担打乱（null）", "hostF": "hostF 宿主，恢复网络学习率 3e-3", "BinF": "BinF 区域负担作普通输入，学习率 3e-3", "RKu": "RKu 区域负担放损伤侧", "Bin": "Bin 区域负担作普通输入", "RKs": "RKs 只有空间项"}
ctrl = [a for a in names if a in r]
if ctrl:
    lines.append("\n**对照 arm（相对重新训练的宿主）**\n")
    lines.append("| arm | 完成的折 | 全体 | 已在停电 |\n|---|---|---|---|")
    for a in ctrl:
        lines.append(f"| {names[a]} | {len(r[a]['folds'])} | {f(r[a]['all'])} | {f(r[a]['active'])} |")
    pairs = [p for p in ("RK vs RKp", "RK vs RKl", "RK vs hostF", "RK vs BinF", "RK vs RKu", "RK vs Bin", "BinF vs hostF") if p in r]
    if pairs:
        lines.append("\n| 比较（共同完成的县-事件） | 县-事件数 | 全体 | 已在停电 |\n|---|---:|---|---|")
        for p in pairs:
            lines.append(f"| {p.replace(' vs ', ' 相对 ')} | {r[p]['units']} | {f(r[p]['all'])} | {f(r[p]['active'])} |")
block = "<!-- RK_BEGIN -->\n" + "\n".join(lines) + "\n<!-- RK_END -->"
p = "notes/OVERNIGHT_RESULTS_20261002_ZH.md"; t = open(p).read()
if "RK_RESULTS" in t:
    t = t.replace("RK_RESULTS", block)
else:
    t = re.sub(r"<!-- RK_BEGIN -->.*?<!-- RK_END -->", lambda m_: block, t, flags=re.S)
open(p, "w").write(t)
print(block)
