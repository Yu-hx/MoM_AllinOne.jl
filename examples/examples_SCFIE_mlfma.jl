## 导入程序包
# using MoM_AllinOne
# using MKL, MKLSparse
using DataFrames, CSV, LaTeXStrings
using CairoMakie, MoM_Basics, MoM_Kernels, MoM_Visualizing  #在using这些包时，代码中的实例会被创建，类似于全局变量了，所以在这后文中可以对这些实例进行修改

## 参数设置
# 设置精度，是否运行时出图等
setPrecision!(Float32)
SimulationParams.SHOWIMAGE = true

# 网格文件
filename = joinpath(@__DIR__, "..", "meshfiles/sphere_600MHz.nas")
meshUnit = :m
## 设置输入频率（Hz）从而修改内部参数
frequency = 6e8

# 积分方程类型
ieT = :CFIE

# 更新基函数类型参数(不推荐更改)
sbfT = :RWG
vbfT = :nothing

# 求解器类型
solverT = :gmres

# 设置 gmres 求解器精度，重启步长(步长越大收敛越快但越耗内存)
rtol = 1e-3
restart = 50

# 源
source = PlaneWave(π / 2, 0, 0.0f0, 1.0f0)

## 观测角度
θs_obs = LinRange{Precision.FT}(-π, π, 721)
ϕs_obs = LinRange{Precision.FT}(0, π / 2, 2)

"""
# 计算脚本  这里将脚本直接复制过来了,方便调试,因为用include方法时,没法打断点调试。
# 原本采用的是这个：
# include(joinpath(@__DIR__, "../src/fast_solver.jl"))
"""
# 更新参数
inputParameters(; frequency=frequency, ieT=ieT)
updateVSBFTParams!(; sbfT=sbfT, vbfT=vbfT)

# 网格文件读取
meshData, εᵣs = getMeshData(filename; meshUnit=meshUnit);

# 基函数生成
ngeo, nbf, geosInfo, bfsInfo = getBFsFromMeshData(meshData; sbfT=sbfT, vbfT=vbfT)

# 设置介电参数
setGeosPermittivity!(geosInfo, 2(1 - 0.0002im))

## 快速算法
# 计算阻抗矩阵（MLFMA计算），注意此处根据基函数在八叉树的位置信息改变了基函数顺序
# Zopt, octree, ZnearCSC  =   getImpedanceOpt(geosInfo, bfsInfo);
nLevels, octree = getOctreeAndReOrderBFs!(geosInfo, bfsInfo; leafCubeEdgel=Precision.FT(0.23Params.λ_0), nInterp=4);

# 叶层
leafLevel = octree.levels[nLevels];
# 计算近场矩阵CSC
ZnearCSC = calZnearCSC(leafLevel, geosInfo, bfsInfo);

# 构建矩阵向量乘积算子
Zopt = MLMFAIterator(ZnearCSC, octree, geosInfo, bfsInfo);

## 根据近场矩阵和八叉树计算 SAI 左预条件
Zprel = sparseApproximateInversePl(ZnearCSC, leafLevel)

# 激励向量
V = getExcitationVector(geosInfo, nbf, source);

# 求解
ICoeff, ch = solve(Zopt, V; solverT=solverT, Pl=Zprel, rtol=rtol, restart=restart);

# RCS
RCSθsϕs, RCSθsϕsdB, RCS, RCSdB = radarCrossSection(θs_obs, ϕs_obs, ICoeff, geosInfo)

"""
***********************************************************************************
"""





## 比较绘图
# 导入feko数据
feko_RCS_file = joinpath(@__DIR__, "../deps/compare_feko/sphere_600MHzRCS.csv")
data_feko = DataFrame(CSV.File(feko_RCS_file, delim=' ', ignorerepeated=true))
RCS_feko = reshape(data_feko[!, "in"], :, 2)

# 绘图保存
fig = farfield2D(θs_obs, 10log10.(RCS_feko), 10log10.(RCS),
    [L"\text{Feko}\;\quad (\phi = \enspace0^{\circ})", L"\text{Feko}\;\quad (\phi = 90^{\circ})"],
    [L"\text{JuMoM} (\phi = \enspace0^{\circ})", L"\text{JuMoM} (\phi = 90^{\circ})"],
    xlabel=L"\theta (^{\circ})", ylabel=L"\text{RCS(dBsm)}", x_unit=:rad, legendposition=:rt)
savedir = joinpath(@__DIR__, "..", "figures")
!ispath(savedir) && mkpath(savedir)
save(joinpath(savedir, "SCFIE_RCS_sphere_600MHz_fast.pdf"), fig)