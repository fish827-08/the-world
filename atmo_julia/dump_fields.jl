# J0 子进程 CLI 包装（**备桥**；主桥是 run_minimal.py 的进程内 juliacall）。
#
# 用法：
#   julia --project=<repo>/atmo_julia/julia_env atmo_julia/dump_fields.jl <outdir> <truncation> <nlayers> <days>
#
# 许可：SpeedyWeather.jl = EUPL-1.2。本脚本只调用未修改的公开包（外部依赖，不 vendor）。

include(joinpath(@__DIR__, "j0_core.jl"))

outdir = ARGS[1]
trunc = parse(Int, ARGS[2])
nlayers = parse(Int, ARGS[3])
days = parse(Int, ARGS[4])

j0_run(outdir, days, trunc, nlayers)
