# J1 核心（第二轮补件）：SpeedyWeather.jl 原生 T32 Held–Suarez 干基准
#
# 协议：`benchmarks/atmo/gate/J1_mapping_freeze.md`（AV 冻结，2026-10-21）
#   * 水平 = **原生 octahedral 约化高斯 T32**（48 环 / 3168 点，20→112 点/环），不映射
#   * 垂直 = 8 层，σ 界面 [0,.05,.14,.26,.40,.55,.70,.85,1]（非均匀阶梯，层中 0.2 恰 200 hPa，
#     供 R-1 追认；垂直配置属 A1 的模型配置自由，AV 冻结件 §5 第 2 行明示）
#   * 初值 = AV 材料化 `iso_w1`（本文件只从 `ic_iso_w1.npz` 读入并施加；初值定义权在裁判侧）
#   * 导出 = 原生展平 (npoints, nlayers)，ps 保持 **ln(Pa)**（不再 exp；AV 映射件升精度后 exp）
#   * 时间 = 状态步长 **Δt = clk.Δt = 144 s**（72,001 步 = 120 天）；2Δt 只是中心差分跨度
#   * 两跑逐位 = 同进程内两次独立 `j1_run`（各新建 model/simulation）
#
# 调用：juliacall 进程内 include → j1_geometry(...) / j1_run(...)
# 许可：SpeedyWeather.jl = EUPL-1.2。只调用未修改的公开包（外部依赖，不 vendor）。
# 写盘：Julia 列主序；Python 侧 np.fromfile(...).reshape((npoints, nlayers), order="F")。

using SpeedyWeather
using Dates
using LinearAlgebra
using SHA

# 确定性：BLAS / FFTW 各 1 线程（两跑逐位的前提；记入 manifest.env）
BLAS.set_num_threads(1)
isdefined(SpeedyWeather, :FFTW) && SpeedyWeather.FFTW.set_num_threads(1)

# 逐位照抄 atmo_core/src/grid.rs::SIGMA_HALF（8 层 sigma 界面，模式顶→地表）
const J1_SIGMA_HALF = [0.0, 0.05, 0.14, 0.26, 0.40, 0.55, 0.70, 0.85, 1.00]
# 层中点按定义在 Float64 里算（模式内部 Float32 存 0.19999999，会让 200 hPa 判据窗口漏层）
const J1_SIGMA_FULL = 0.5 .* (J1_SIGMA_HALF[1:(end - 1)] .+ J1_SIGMA_HALF[2:end])
const J1_TRUNCATION = 32
const J1_NLAYERS = 8

# 原生环坐标（与 variables.grid.*.data 的展平顺序严格一致：环升序，环内 lon 0→360-Δ）
function _ring_coords(g)
    npoints = RingGrids.get_npoints(g)
    latd = RingGrids.get_latd(g)
    lat = Vector{Float64}(undef, npoints)
    lon = Vector{Float64}(undef, npoints)
    for j in 1:length(g.rings)
        r = g.rings[j]
        nlon = length(r)
        for (i, idx) in enumerate(r)
            lat[idx] = Float64(latd[j])
            lon[idx] = 360.0 * (i - 1) / nlon
        end
    end
    return lat, lon
end

function _build_sim(; NF::Type{<:AbstractFloat}=Float32, truncation::Integer=J1_TRUNCATION,
                    nlayers::Integer=J1_NLAYERS, dt_clock_s::Integer=144)
    # 默认 Grid = OctahedralGaussianGrid（原生约化高斯，T32 ⇒ 48 环 / 3168 点）
    sg = SpectralGrid(NF=NF, truncation=truncation, nlayers=nlayers)
    vc = SigmaCoordinates(sg, J1_SIGMA_HALF)
    geom = Geometry(sg; vertical_coordinates=vc)
    planet = Earth(sg; rotation=7.292e-5, gravity=9.81, radius=6.371e6)
    atm = EarthDryAtmosphere(sg; R_dry=287.0, heat_capacity=1004.0)
    dt_at_T32 = Second(round(Int, dt_clock_s * 32 / 32))
    ts = Leapfrog(sg; Δt_at_T32=dt_at_T32, adjust_with_output=false)

    model = PrimitiveDryModel(; spectral_grid=sg, geometry=geom, planet=planet, atmosphere=atm,
        dynamics_only=true,                 # 关物理参数化（辐射/对流/云…），动力核照跑
        orography=NoOrography(sg),
        land_sea_mask=AquaPlanetMask(sg),
        ocean=AquaPlanet(sg),
        land=nothing,
        forcing=HeldSuarez(sg),             # HS94 温度松弛（显式开；默认是 nothing）
        drag=LinearDrag(sg),                # HS94 线性边界层拖曳（默认是 SpeedLimitDrag）
        time_stepping=ts,
    )
    sim = Simulation(model)
    return sim, sg, model
end

# ---------------------------------------------------------------- 几何（供 AV 材料化 IC）

"""
    j1_geometry(outdir; nlayers=8)

只落网格几何（lat/lon/sigma 与一个零场快照），**无物理结果**；供 AV 侧
`benchmarks.atmo.jia_remap --ic iso_w1` 材料化初值。返回 J1_GEOM 摘要行。
"""
function j1_geometry(outdir::AbstractString; nlayers::Integer=J1_NLAYERS)
    mkpath(outdir)
    sim, sg, model = _build_sim(; nlayers=nlayers)
    g = sg.grid
    npoints = RingGrids.get_npoints(g)
    nlev = sg.nlayers
    lat, lon = _ring_coords(g)

    write(joinpath(outdir, "lat.bin"), lat)               # Float64，原生顺序
    write(joinpath(outdir, "lon.bin"), lon)
    write(joinpath(outdir, "sigma_full.bin"), J1_SIGMA_FULL)
    write(joinpath(outdir, "sigma_half.bin"), J1_SIGMA_HALF)

    tag = "00001"
    write(joinpath(outdir, "snap_$(tag)_u.bin"), zeros(Float32, npoints, nlev))
    write(joinpath(outdir, "snap_$(tag)_v.bin"), zeros(Float32, npoints, nlev))
    write(joinpath(outdir, "snap_$(tag)_T.bin"), zeros(Float32, npoints, nlev))
    write(joinpath(outdir, "snap_$(tag)_ps.bin"), fill(log(100000.0), npoints))

    counts = [length(r) for r in g.rings]
    forcing = model.forcing
    drag = model.drag
    summary = string(
        "J1_GEOM npoints=", npoints, " nlayers=", nlev,
        " nlat=", 2 * g.nlat_half, " nlat_half=", g.nlat_half,
        " nring=", length(counts), " ring_min=", minimum(counts), " ring_max=", maximum(counts),
        " truncation=", sg.truncation, " NF=", sg.NF,
        " lat_reach_deg=", maximum(abs.(lat)),
        " sigma=", join(round.(J1_SIGMA_FULL, digits = 6), ","),
        " julia=", VERSION, " speedy=", pkgversion(SpeedyWeather),
        " nthreads=", Threads.nthreads(),
        " omega=", Float64(model.planet.rotation), " g=", Float64(model.planet.gravity),
        " radius=", Float64(model.planet.radius),
        " R=", Float64(model.atmosphere.R_dry), " cp=", Float64(model.atmosphere.heat_capacity),
        " p0=", Float64(model.atmosphere.reference_pressure),
        " delta_Ty=", Float64(forcing.ΔTy), " delta_theta_z=", Float64(forcing.Δθz),
        " sigma_b=", Float64(forcing.σb),
        " ka=", 1 / Second(forcing.relax_time_slow).value,
        " ks=", 1 / Second(forcing.relax_time_fast).value,
        " kf=", 1 / Second(drag.time_scale).value,
        " drag_sigma_b=", Float64(drag.σb),
    )
    println(summary)
    return summary
end

# ---------------------------------------------------------------- 初值（AV 材料化，A1 施加）

# IC 以<float32 小端原始字节>旁挂：ic_iso_w1_{u,v,T,ps}.bin（Python 侧从 AV 材料化的 npz
# 逐字节 dump）。不用 PythonCall 读 npz —— `z["u"]` 返回 Py(nothing) 会让 Array 构造失败。
function _read_ic(ic_path::AbstractString, npoints::Integer, nlayers::Integer,
                  NF::Type{<:AbstractFloat}, expect_sha::Union{Nothing, AbstractString})
    base = endswith(ic_path, ".npz") ? ic_path[1:(end - 4)] : ic_path
    keys = ("u", "v", "T", "ps")
    paths = [base * "_" * k * ".bin" for k in keys]
    for p in paths
        isfile(p) || error("IC 二进制缺件 $p（先跑 run_hs_julia.py --stage ic）")
    end
    raw = read.(paths)
    if expect_sha !== nothing
        got = bytes2hex(sha256(reduce(vcat, raw)))
        got == lowercase(expect_sha) || error("IC sha256 不符: 实测 $got ≠ manifest $expect_sha")
    end
    function arr(i, shape)
        n = prod(shape)
        length(raw[i]) == 4n || error("IC $(paths[i]) 字节数 $(length(raw[i])) != 4×$n")
        return Array{NF}(reshape(reinterpret(Float32, raw[i]), shape))
    end
    u = arr(1, (npoints, nlayers))
    v = arr(2, (npoints, nlayers))
    T = arr(3, (npoints, nlayers))
    ps = arr(4, (npoints,))
    return u, v, T, ps
end

function _apply_ic!(sim, sg, ic_path::AbstractString;
                    expect_sha::Union{Nothing, AbstractString} = nothing)
    NF = sg.NF
    nlayers = sg.nlayers
    g = sg.grid
    npoints = RingGrids.get_npoints(g)
    u, v, T, ps = _read_ic(ic_path, npoints, nlayers, NF, expect_sha)
    # 两个 leapfrog 步都写（否则只写 step1 会被 step2 覆盖）
    for s in 1:2
        fu = RingGrids.Field(NF, g, nlayers)
        fv = RingGrids.Field(NF, g, nlayers)
        fT = RingGrids.Field(NF, g, nlayers)
        fps = RingGrids.Field(NF, g)
        fu.data .= u
        fv.data .= v
        fT.data .= T
        fps.data .= ps
        set!(sim; u=fu, v=fv, temperature=fT, pressure=fps, step=s)
    end
    return nothing
end

# ---------------------------------------------------------------- 快照

function _write_snapshot(outdir, idx, sim, npoints, nlev)
    prog = sim.variables.grid
    step = size(prog.u.data, 3)            # Leapfrog 当前步
    U = Array(view(prog.u.data, :, :, step))
    V = Array(view(prog.v.data, :, :, step))
    T = Array(view(prog.temperature.data, :, :, step))
    ps = Array(view(prog.pressure.data, :, step))   # ln(Pa)
    tag = lpad(idx, 5, '0')
    write(joinpath(outdir, "snap_$(tag)_u.bin"), U)
    write(joinpath(outdir, "snap_$(tag)_v.bin"), V)
    write(joinpath(outdir, "snap_$(tag)_T.bin"), T)
    write(joinpath(outdir, "snap_$(tag)_ps.bin"), ps)
    return (minimum(T), maximum(T), maximum(abs.(U)), minimum(exp.(ps)), maximum(exp.(ps)))
end

"""
    j1_run(outdir; days=120, dt_clock_s=144, nlayers=8, ic_path, ic_sha256, NF=Float32,
           snap_start_day=30.0, snap_every_day=2.0)

跑一次原生 T32 Held–Suarez 干基准，返回 J1_DONE 摘要行。每次调用新建 model/simulation。
"""
function j1_run(
        outdir::AbstractString;
        days::Integer = 120,
        dt_clock_s::Integer = 144,
        nlayers::Integer = J1_NLAYERS,
        ic_path::AbstractString,
        ic_sha256::Union{Nothing, AbstractString} = nothing,
        dtype::AbstractString = "float32",
        snap_start_day::Real = 30.0,
        snap_every_day::Real = 2.0,
    )
    NF = dtype == "float64" ? Float64 : Float32
    mkpath(outdir)
    sim, sg, model = _build_sim(; NF=NF, nlayers=nlayers, dt_clock_s=dt_clock_s)
    g = sg.grid
    npoints = RingGrids.get_npoints(g)
    nlev = sg.nlayers
    lat, lon = _ring_coords(g)

    write(joinpath(outdir, "lat.bin"), lat)
    write(joinpath(outdir, "lon.bin"), lon)
    write(joinpath(outdir, "sigma_full.bin"), J1_SIGMA_FULL)
    write(joinpath(outdir, "sigma_half.bin"), J1_SIGMA_HALF)

    _apply_ic!(sim, sg, ic_path; expect_sha = ic_sha256)

    # 时间循环（不用 run!：需要在循环内按时刻落快照；Leapfrog 自旋起步只跑一次）
    initialize!(sim; period = Day(days), output = false)
    clk = sim.variables.prognostic.clock
    Δt_s = Float64(clk.Δt.value) / 1000
    steps_per_day = 86400 / Δt_s
    start_step = round(Int, snap_start_day * steps_per_day)
    every_step = round(Int, snap_every_day * steps_per_day)

    next_target = start_step
    idx = 0
    last = (NaN, NaN, NaN, NaN, NaN)
    t0 = time()
    for _ in 1:clk.n_steps
        SpeedyWeather.time_step!(sim)
        if clk.time_step_counter >= next_target
            idx += 1
            last = _write_snapshot(outdir, idx, sim, npoints, nlev)
            next_target += every_step
        end
    end
    wall = time() - t0

    forcing = model.forcing
    drag = model.drag
    summary = string(
        "J1_DONE npoints=", npoints, " nlayers=", nlev,
        " nlat=", 2 * g.nlat_half, " nlat_half=", g.nlat_half,
        " truncation=", sg.truncation, " NF=", sg.NF,
        " n_steps=", clk.n_steps, " n_snaps=", idx,
        " dt_clock_s=", Δt_s, " dt_step_s=", Δt_s, " dt_center_s=", 2 * Δt_s,
        " julia=", VERSION, " speedy=", pkgversion(SpeedyWeather),
        " nthreads=", Threads.nthreads(),
        " sigma=", join(round.(J1_SIGMA_FULL, digits = 6), ","),
        " omega=", Float64(model.planet.rotation), " g=", Float64(model.planet.gravity),
        " radius=", Float64(model.planet.radius),
        " R=", Float64(model.atmosphere.R_dry), " cp=", Float64(model.atmosphere.heat_capacity),
        " p0=", Float64(model.atmosphere.reference_pressure),
        " delta_Ty=", Float64(forcing.ΔTy), " delta_theta_z=", Float64(forcing.Δθz),
        " sigma_b=", Float64(forcing.σb),
        " ka=", 1 / Second(forcing.relax_time_slow).value,
        " ks=", 1 / Second(forcing.relax_time_fast).value,
        " kf=", 1 / Second(drag.time_scale).value,
        " drag_sigma_b=", Float64(drag.σb),
        " wall_s=", round(wall, digits = 2),
        " T_range=", last[1], ",", last[2], " u_absmax=", last[3],
        " ps_range=", last[4], ",", last[5],
    )
    println(summary)
    return summary
end
