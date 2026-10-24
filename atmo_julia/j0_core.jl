# J0 核心：SpeedyWeather.jl 最小干动力跑 + float32 二进制导出
#
# 两种调用方式共用本文件：
#   ① 进程内（主桥）：Python import juliacall → include(j0_core.jl) → j0_run(...)
#   ② 子进程（备桥）：dump_fields.jl CLI 包装（juliacall 不可用时）
#
# 许可：SpeedyWeather.jl = EUPL-1.2。本脚本只调用未修改的公开包（外部依赖，不 vendor）。
#
# 说明：truncation=32 / 8 层 ⇒ octahedral Gaussian 3168 点（≈旧 T21 档）。
#       dynamics_only=true ⇒ 纯干动力（无辐射/对流/云）；资产全部用无资产配置（网络不可依赖）。

using SpeedyWeather

function j0_run(outdir::AbstractString, days::Integer, truncation::Integer, nlayers::Integer)
    mkpath(outdir)

    grid = SpectralGrid(truncation=truncation, nlayers=nlayers)
    model = PrimitiveDryModel(; spectral_grid=grid,
        dynamics_only=true,                    # 纯干动力（不开辐射/对流/云）
        orography=NoOrography(grid),           # 不读地形资产
        land_sea_mask=AquaPlanetMask(grid),    # 全海
        ocean=AquaPlanet(grid),                # 常量 SST 分布（不读气候态文件）
        land=nothing)                          # 无陆地 ⇒ 不读土壤温度资产

    sim = initialize!(model)
    t0 = time()
    run!(sim, period=Day(days))
    wall = time() - t0

    u = sim.variables.grid.u.data            # (npoints, nlayers, 2)
    v = sim.variables.grid.v.data
    T = sim.variables.grid.temperature.data
    ps_log = sim.variables.grid.pressure.data   # (npoints, 2) —— **这是 ln(p_s)**（源码：units="log(Pa)"）
    ps = exp.(ps_log)                           # 转成 Pa 再导出

    g = sim.model.spectral_grid.grid
    npoints = RingGrids.get_npoints(g)
    latd = RingGrids.get_latd(g)
    lat = zeros(Float32, npoints)
    lon = zeros(Float32, npoints)
    for j in 1:(2*g.nlat_half)
        r = g.rings[j]
        nlon = length(r)
        for (i, idx) in enumerate(r)
            lat[idx] = latd[j]
            lon[idx] = 360 * (i - 1) / nlon
        end
    end

    k = size(u, 3)
    write(joinpath(outdir, "u.bin"), Float32.(u[:, :, k]))
    write(joinpath(outdir, "v.bin"), Float32.(v[:, :, k]))
    write(joinpath(outdir, "T.bin"), Float32.(T[:, :, k]))
    write(joinpath(outdir, "ps.bin"), Float32.(ps[:, k]))   # 已转 Pa
    write(joinpath(outdir, "lat.bin"), lat)
    write(joinpath(outdir, "lon.bin"), lon)

    summary = string("J0_DONE npoints=", npoints, " nlayers=", size(u, 2), " wall=", round(wall, digits=2),
        " T=[", minimum(T[:, :, k]), ",", maximum(T[:, :, k]), "]",
        " u=[", minimum(u[:, :, k]), ",", maximum(u[:, :, k]), "]",
        " ps_Pa=[", minimum(ps[:, k]), ",", maximum(ps[:, k]), "]",
        " finite=", all(isfinite, T[:, :, k]) && all(isfinite, ps[:, k]),
        " julia=", VERSION, " speedy=", pkgversion(SpeedyWeather),
        " sigma=", join(round.(sim.model.geometry.σ_levels_full, digits=6), ","))
    println(summary)
    return summary
end
