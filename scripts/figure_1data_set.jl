using PyPlot



path_functions="/home/genis/wm_mice/scripts/functions/"
path_figures="/home/genis/wm_mice/figures/"
path_results="/home/genis/wm_mice/results/"
include(path_functions*"functions_wm_mice.jl")


Ntrials=Int(1e3)
filename_load=path_results*"minimize_sigma_x0_betal_pdwdw_pbiasbias_Ntrials"*string(Ntrials)*".jld"
data=load(filename_load)

LlOriginal=data["LlOriginal"]
index=findall(!isnan,data["Ll"])

Ll=data["Ll"][index]

ParamFit=data["ParamFit"][index,:]
XInitial=data["XInitial"][index,:]
PiInitial=data["PiInitial"][index,:]
TInitialAll=data["TInitialAll"][index,:,:]
ParamOriginal=vcat([data["T"][1,1] ],[ data["T"][2,2] ],x)

fig,ax2= subplots(1,2,figsize=(10,3))
ax2[1].hist(Ll,bins=100)
ax2[1].plot([data["LlOriginal"],data["LlOriginal"]],[0,10],"r-")
index2=findall(x-> x<517 ,Ll)
ParamFit=ParamFit[index2,:]
NGoodFits=length(ParamFit[:,1])
Nparam=length(ParamFit[1,:])
for iparam in 1:Nparam
    ax2[2].plot(ParamFit[:,iparam],iparam.+0.5.*rand(NGoodFits),".")
    ax2[2].plot([ParamOriginal[iparam],ParamOriginal[iparam]],iparam.+[-0.25,0.25],"k-")
end
