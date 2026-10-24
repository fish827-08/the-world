"""AV 冻结判据注册表。

纪律：判据先提交（commit 时间戳为证）后跑；阈值来自文献/基准，不按结果回调；
每条判据必须带 citation。数值含义与量纲见各条 desc/unit。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Criterion:
    cid: str
    desc: str
    mandatory: bool
    citation: str
    limit_lo: Optional[float] = None
    limit_hi: Optional[float] = None
    unit: str = ""


C_HS = (
    "Held & Suarez (1994), BAMS 75(10):1825-1841。"
    "各带为按该文统计描述冻结的宽松 sanity 带（细判另走 reference.py 只读对照）。"
)
C_AQ = (
    "Neale & Hoskins (2000), Atmos. Sci. Lett. 1:101-108（aquaplanet 标准试验框架；"
    "本实现用 slab 海洋而非规定 SST，属已声明偏差）；ITCZ 降水量级参照 GPCP 热带洋面气候态"
    "（Adler et al. 2003, J. Hydrometeorol. 4:1147-1167，热带洋面雨带峰值 ~8-12 mm/d）。"
)
C_CONS = (
    "质量漂移带参照 DCMIP 2016 动力核测试文档（Ullrich et al., 2016, GMD 附件：守恒诊断）；"
    "能量闭合量级参照 Trenberth et al. (2009), BAMS 90:311-323（TOA 净通量观测不确定度 ~1 W/m^2）。"
)
C_CFL = "Courant-Friedrichs-Lewy (1928); 综述见 Durran (1999), Numerical Methods for Wave Equations in Geophysical Fluid Dynamics。"
C_DET = "the-world 仓库级不变量：同配置同 seed 两次运行逐位一致（内部工程指南 关键约束 1 的确定性条款）。"

C_REG = "the-world 写死账本（内部技术文档/技术-写死账本-20261019.md）H4：pr_anchor/g_wet/g_dry 手画带家族为已知写死形态。"

C_SOLAR = (
    "atmo/INTERFACE.md FROZEN v1.0.2 §8.5/§11 OPEN-2（太阳几何归 A1，ac.solar 为唯一真源；"
    "A2 atmo/physics/solar.py 仅作列模式与对拍参考）；解析公式 Hartmann, "
    "Global Physical Climatology 2e §2.3。"
)

C_RCE = (
    "atmo/INTERFACE.md §11 附加冻结条件 2（A2 报告 D1 采纳）：单柱 RCE 判据 = "
    "系统能量闭合 |TOA - dE/dt| < 5 W/m^2 + 准平衡 |dTs/dt| < 0.05 K/d；"
    "单柱缺大尺度下沉，不要求 TOA≈0（AV 判据注册表按裁定写明此口径）。"
)


CRITERIA = {
    # ---- 守恒 / 稳定性（两套件通用；CONS-ENERGY 仅湿套件适用）----
    "CONS-MASS": Criterion(
        "CONS-MASS",
        "干质量守恒：全期 ps 全球面积加权平均相对 t0 的最大漂移 |dM|/M0",
        True, C_CONS, limit_hi=1e-6, unit="1"),
    "CONS-WATER": Criterion(
        "CONS-WATER",
        "全球水账闭合 |E - P - dW/dt| / P_mean（E/P/dW-dt 均 SI kg m-2 s-1，全球面积加权、时间平均）",
        True, C_CONS, limit_hi=1e-3, unit="1"),
    "CONS-ENERGY": Criterion(
        "CONS-ENERGY",
        "气柱+slab 能量闭合：|mean(SW_toa_net - LW_out) - C_slab * dSST/dt|（后 50% 时段全球面积加权）",
        True, C_CONS, limit_hi=1.0, unit="W m-2"),
    "CONS-CFL": Criterion(
        "CONS-CFL",
        "AV 独立重算的最大平流 CFL = dt*(|u|/dx + |v|/dy)（快照场逐快照取 max；"
        "排除极区各 2 行——经纬网格极点 dx→0 的已知格点效应由 CONS-POLAR-* 另管；"
        "与 meta 报告值取大者）",
        True, C_CFL, limit_hi=1.0, unit="1"),
    "CONS-NAN": Criterion(
        "CONS-NAN", "全部状态/通量场无 NaN/Inf", True, C_CONS, unit="-"),
    "CONS-POLAR-AMP": Criterion(
        "CONS-POLAR-AMP",
        "极区稳定（振幅）：极区各 2 行最大|风| / 其余区域最大|风|，全期 max（>1 即极区伪放大）",
        True, C_CONS, limit_hi=3.0, unit="1"),
    "CONS-POLAR-TREND": Criterion(
        "CONS-POLAR-TREND",
        "极区稳定（趋势）：极区最大|风|序列末 20% 时段线性斜率（杜绝单调增长的极区伪放大）",
        True, C_CONS, limit_hi=0.01, unit="m s-1 day-1"),
    "CONS-DET": Criterion(
        "CONS-DET", "同配置同 seed 第二次运行：全部快照与 meta 逐位一致",
        True, C_DET, unit="-"),

    # ---- Held-Suarez 干动力 ----
    "HS-FORCING": Criterion(
        "HS-FORCING",
        "meta.forcing 与 HS94 标准强迫参数逐项一致（相对差 <=1%；缺失/超差键计数）",
        True, C_HS, limit_hi=0.0, unit="count"),
    "HS-JET-U": Criterion(
        "HS-JET-U", "时均纬向平均 u 在 200-400 hPa 窗口内的最大值",
        True, C_HS, limit_lo=15.0, limit_hi=40.0, unit="m s-1"),
    "HS-JET-LAT": Criterion(
        "HS-JET-LAT", "上项急流最大值所在纬度（|lat|）",
        True, C_HS, limit_lo=20.0, limit_hi=60.0, unit="deg"),
    "HS-SFC-U": Criterion(
        "HS-SFC-U", "时均纬向平均 u 在 sigma>=0.9 且 |lat|<=30 的区域平均（近地面热带东风）",
        True, C_HS, limit_lo=-8.0, limit_hi=2.0, unit="m s-1"),
    "HS-T-TROP-SFC": Criterion(
        "HS-T-TROP-SFC", "时均纬向平均 T 在 sigma>=0.9 且 |lat|<=15 的区域平均",
        True, C_HS, limit_lo=290.0, limit_hi=320.0, unit="K"),
    "HS-T-POLE-SFC": Criterion(
        "HS-T-POLE-SFC", "时均纬向平均 T 在 sigma>=0.9 且 |lat|>=75 的区域平均",
        True, C_HS, limit_lo=225.0, limit_hi=265.0, unit="K"),
    "HS-T-TROP-UP": Criterion(
        "HS-T-TROP-UP", "时均纬向平均 T 在 200-250 hPa（或 sigma 0.20-0.25）且 |lat|<=15 的区域平均",
        True, C_HS, limit_lo=185.0, limit_hi=230.0, unit="K"),
    "HS-EKE-ML": Criterion(
        "HS-EKE-ML", "瞬变涡动动能（对时-纬平均偏差）在 300-700 hPa、lat 40-60 的平均",
        True, C_HS, limit_lo=1.0, unit="m2 s-2"),
    "HS-EKE-RATIO": Criterion(
        "HS-EKE-RATIO", "EKE(40-60deg) / EKE(|lat|<=15deg)（同 300-700 hPa 窗口；中纬斜压活动占优）",
        True, C_HS, limit_lo=1.5, unit="1"),

    # ---- 水行星湿基准 ----
    "AQ-FORCING": Criterion(
        "AQ-FORCING",
        "meta 必须登记辐射方案描述与 slab 热容（可追溯性门：缺失键计数）",
        True, C_AQ, limit_hi=0.0, unit="count"),
    "AQ-ITCZ-LAT": Criterion(
        "AQ-ITCZ-LAT",
        "时均纬向平均降水在 |lat|<=20 内最大值纬度（仅 obliquity=0 对称配置判；否则记 N/A）",
        True, C_AQ, limit_lo=-5.0, limit_hi=5.0, unit="deg"),
    "AQ-ITCZ-P": Criterion(
        "AQ-ITCZ-P", "时均纬向平均降水热带峰值",
        True, C_AQ, limit_lo=4.0, limit_hi=20.0, unit="mm d-1"),
    "AQ-EP-SUBTROP": Criterion(
        "AQ-EP-SUBTROP", "E-P 在副热带 20-35deg 的时间-区域平均（应蒸发>降水）",
        True, C_AQ, limit_lo=0.0, unit="mm d-1"),
    "AQ-TOA": Criterion(
        "AQ-TOA", "后 50% 时段全球面积加权 TOA 净通量 mean(SW_toa_net - LW_out)",
        True, C_AQ, limit_lo=-5.0, limit_hi=5.0, unit="W m-2"),
    "AQ-SST-TREND": Criterion(
        "AQ-SST-TREND", "全球面积加权 SST 序列后 50% 时段线性斜率（准平衡门）",
        True, C_AQ, limit_hi=0.01, unit="K day-1"),
    "AQ-HEMI-DIFF": Criterion(
        "AQ-HEMI-DIFF",
        "时均降水半球差 |P_NH(0-60) - P_SH(0-60)|（仅 obliquity=0 判；否则记 N/A）",
        True, C_AQ, limit_hi=2.0, unit="mm d-1"),

    # ---- 太阳几何对拍与 sanity（INTERFACE §11 OPEN-2 指定 AV 增加）----
    "SOLAR-XCHECK-COSZ": Criterion(
        "SOLAR-XCHECK-COSZ",
        "A1 ac.solar 的 cosz 与 A2 solar.py 参考实现（同 δ/H 采样）的最大绝对差",
        True, C_SOLAR, limit_hi=1e-6, unit="1"),
    "SOLAR-XCHECK-TOA": Criterion(
        "SOLAR-XCHECK-TOA",
        "A1 ac.solar 的 toa_sw 与 A2 参考实现的最大相对差"
        "（归一到 S0*(a/r)^2，即同日地距离因子下的直射通量差）",
        True, C_SOLAR, limit_hi=1e-6, unit="1"),
    "SOLAR-XCHECK-DEC": Criterion(
        "SOLAR-XCHECK-DEC",
        "A1 subsolar_lat_deg 与 A2 解析赤纬 asin(sin eps * sin lam) 的最大绝对差"
        "（相位约定按 INTERFACE §9.3：t=equinox_offset_s 为北半球春分）",
        True, C_SOLAR, limit_hi=1e-6, unit="deg"),
    "SOLAR-COSZ-RANGE": Criterion(
        "SOLAR-COSZ-RANGE",
        "sanity：采样全体 cosz 落在 [0, 1]（含 1e-12 容差），越界点计数",
        True, C_SOLAR, limit_hi=0.0, unit="count"),
    "SOLAR-GLOBAL-MEAN": Criterion(
        "SOLAR-GLOBAL-MEAN",
        "sanity：全年采样瞬时 TOA 全球面积平均相对 |mean - S0/4| / (S0/4)"
        "（几何恒等式：任一时刻全球平均 = S0*(a/r)^2/4，年均值 ≈ S0/4）",
        True, C_SOLAR, limit_hi=0.02, unit="1"),
    "SOLAR-OBLIQ0-SYMM": Criterion(
        "SOLAR-OBLIQ0-SYMM",
        "sanity：obliquity=0 时 cosz 关于赤道对称 |cosz(lat) - cosz(-lat)| 最大值",
        True, C_SOLAR, limit_hi=1e-6, unit="1"),
    "SOLAR-POLAR-NIGHT": Criterion(
        "SOLAR-POLAR-NIGHT",
        "sanity：obliquity>0 时极圈纬度（|lat|>90-eps_obl）存在极夜采样点（toa_sw=0 计数 > 0）",
        True, C_SOLAR, limit_lo=1.0, unit="count"),

    # ---- A2 单柱 RCE（INTERFACE §11 附加 2）----
    "RCE-COL-ENERGY": Criterion(
        "RCE-COL-ENERGY",
        "单柱 RCE 系统能量闭合 |TOA_net - dE/dt|（时均，E = 气柱 + 地表储能）",
        True, C_RCE, limit_hi=5.0, unit="W m-2"),
    "RCE-COL-EQUIL": Criterion(
        "RCE-COL-EQUIL",
        "单柱 RCE 准平衡 |dTs/dt|（末 50% 时段线性斜率）；单柱不要求 TOA≈0（缺大尺度下沉），"
        "判准平衡而非绝对 TOA",
        True, C_RCE, limit_hi=0.05, unit="K day-1"),
}


def get(cid: str) -> Criterion:
    return CRITERIA[cid]


def check_value(cid: str, x: float) -> bool:
    """带外判（含端点）。limit 为 None 表示该侧不设限。"""
    c = CRITERIA[cid]
    ok_lo = c.limit_lo is None or x >= c.limit_lo
    ok_hi = c.limit_hi is None or x <= c.limit_hi
    return bool(ok_lo and ok_hi)
