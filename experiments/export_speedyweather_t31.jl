# SpeedyWeather T31 地球气候表导出（模板 —— 必须在 Linux 云端机上跑）
#
# 🔴 本文件**未在本机验证**（本机无 Julia；WSL 被安全策略拉黑）。
#    它是「拿到云端机后照着跑」的作业单，不是可交付的成品。版本适配点都用 `# ADAPT` 标出。
#
# 环境（派工指定；conda-forge 的 julia 比直下 274 MB 快）
#   mamba create -n gcm -c conda-forge julia
#   mamba activate gcm
#   julia -e 'using Pkg; Pkg.add("SpeedyWeather"); Pkg.add("RingGrids")'
#
# 跑法
#   julia export_speedyweather_t31.jl > t31_earth_raw.csv
#   # 回本机：python experiments/make_gcm_table.py --csv t31_earth_raw.csv \
#   #            --out the-world-data/gcm/t31_earth_v0/t31_earth_v0.npz \
#   #            --source "SpeedyWeather.jl <ver> T31 10L 3yr spin-up + 1yr monthly" --months 12
#
# 输出 CSV 列（对齐 world.climate.GCM_TABLE_FORMAT 的上游）：
#   lat,lon,month,temperature_c,precip_mm_yr,u,v
#   * lat/lon 单位度；lon 用 0..360（RingGrids 默认即 0..360）
#   * month = 1..12（12 月气候态）；要年平表就去掉 month 列（后处理取均值即可）
#   * precip_mm_yr 用 mm/yr：SpeedyWeather 的降水率是 kg/m²/s ⇒ ×86400×365.25
#   * u/v = 最低层（近地面）风，m/s；v 可先不给（世界侧 wind() 允许 v 缺省）

using SpeedyWeather
using RingGrids

const TRUNC   = 31          # 派工口径：T31（≈400 km）
const NLAYERS = 10          # 派工口径：10 层
const SPINUP_DAYS = 365 * 3 # 3 年自旋
const MONTH_DAYS  = 30      # 简化月长（气候态用途，不做真实日历）

spectral_grid = SpectralGrid(trunc = TRUNC, nlayers = NLAYERS)
model = PrimitiveWetModel(spectral_grid)
simulation = initialize!(model)

# ---- 1) 自旋 ----
run!(simulation, period = Day(SPINUP_DAYS))

# ---- 2) 逐月输出（每月跑 MONTH_DAYS 天，取月末瞬时场）----
# ADAPT：不同版本的 diagnostic_variables 路径不同（v0.19–v0.22 实测过 grid.temperature /
#        grid.u_grid / grid.precipitation_rate）。云端机上先 `dump(simulation.diagnostic_variables.grid)`
#        确认字段名，再改下面三行。
println("lat,lon,month,temperature_c,precip_mm_yr,u,v")
for month in 1:12
    run!(simulation, period = Day(MONTH_DAYS))

    grid = simulation.diagnostic_variables.grid
    T = grid.temperature[1, :]          # ADAPT: 最低层气温（K）→ 下面转 ℃
    u = grid.u_grid[1, :]               # ADAPT: 最低层纬向风
    v = grid.v_grid[1, :]               # ADAPT: 最低层经向风
    P = grid.precipitation_rate[1, :]   # ADAPT: kg/m²/s

    lats = RingGrids.get_latd(T)
    lons = RingGrids.get_lond(T)

    for i in eachindex(lats)
        t_c = T[i] - 273.15
        p_mm = P[i] * 86400 * 365.25
        println(round(lats[i], digits = 4), ",", round(lons[i], digits = 4), ",",
                month, ",", round(t_c, digits = 4), ",", round(p_mm, digits = 4), ",",
                round(u[i], digits = 4), ",", round(v[i], digits = 4))
    end
end

# ⚠️ 云端机上必须核对的三件事（写进汇报，不许猜）：
# 1) 网格类型：默认可能是 OctahedralGaussianGrid（每环经度点数不同）⇒ 本脚本逐点输出
#    (lat,lon)，`make_gcm_table.py` 靠 `np.unique(lat/lon)` 建规则网格会失败。
#    解决：跑之前加 `spectral_grid = SpectralGrid(trunc=TRUNC, nlayers=NLAYERS,
#    grid=FullGaussianGrid)`，或先 `RingGrids.interpolate` 到规则 1°×1° 再输出。
# 2) 单位：温度 K、降水 kg/m²/s、风 m/s —— 三个都要现场核（`dump` 或文档）。
# 3) 自旋年数：3 年只是下限；判据是"全球平均温度年际漂移 < 0.1 K"（加跑 1 年对比）。
