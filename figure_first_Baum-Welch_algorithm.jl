using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches

const epsilon=1e-8

using Distributions
using HMMBase
using Pandas
###### definitions ####
fig,ax=subplots(1,4)
fig2,ax2=subplots(1,4)


Ntrials=[100,1000,10000]
for itrial in 1:length(Ntrials)
    #filename_save="/home/genis/wm_mice/synthetic_data/HmmBaseFitHmmCategorical_Ntrials"*string(Ntrials[itrial])*".jld"

    filename_save="/home/genis/wm_mice/synthetic_data/FitHmmCategorical_Ntrials"*string(Ntrials[itrial])*".jld"
    data=JLD.load(filename_save)
    TBest=data["TBest"]
    PBest=data["PBest"]
    PiBest=data["PiBest"]
    LlBest=data["LlBest"]
    LlOriginal=data["LlOriginal"]
    T=data["T"]
    P=data["P"]
    NDataSets=length(LlOriginal)

    ax[itrial].plot(TBest[:,1,1],1.0 .+rand(NDataSets)./3,"k.")
    ax[itrial].plot(TBest[:,2,2],2.0 .+rand(NDataSets)./3,"k.")
    ax[itrial].plot(PBest[:,1,1],3.0 .+rand(NDataSets)./3,"k.")
    ax[itrial].plot(PBest[:,2,2],4.0 .+rand(NDataSets)./3,"k.")



    ax[itrial].plot([T[1,1],T[1,1]],[1,3],"r-")
    ax[itrial].plot([T[2,2],T[2,2]],[1,3],"r-")
    # ax[itrial].plot([T[1,2],T[1,2]],[1,3],"r-")
    # ax[itrial].plot([T[2,1],T[2,1]],[1,3],"r-")

    ax[itrial].plot([P[1,1],P[1,1]],[3,5],"b--")
    ax[itrial].plot([P[2,2],P[2,2]],[3,5],"b--")
    # ax[itrial].plot([P[1,2],P[1,2]],[3,5],"b--")
    # ax[itrial].plot([P[2,1],P[2,1]],[3,5],"b--")

    ax2[itrial].plot(LlOriginal,"r.")
    ax2[itrial].plot(LlBest,"k.")
end
