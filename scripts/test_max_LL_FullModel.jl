using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches

path_functions="/home/genis/wm_mice/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
include(path_functions*"functions_mle.jl")


PDwDw=0.9
PBiasBias=0.1
PrDw=0.9
PrBias=0.3




consts=["mu_k","c2","c4","mu_b","beta_w","tau_w","tau_l"]
y=[    0.3,  1.2, 1.0, -0.05,     3.0,     10,     10]

args=["sigma","x0","beta_l"]
x=[0.3,0.15,-1.0]

param=make_dict(args,x,consts,y)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e3)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)


T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]
PiInitialOriginal=[0 1]
#
choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,T,args,x,consts,y)

############## sanity checks data ################################3
#pr,pstate=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)
#ll=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)

#xx=zeros(typeof(SIGMA[1]),1)
# indexDw=findall(x->x==1,state[1:Ntrials])
# indexBias=findall(x->x==0,state[1:Ntrials])
# println(mean((choices[indexDw].+1)./2)," ",mean((choices[indexBias].+1)./2))
# figure()
#
# plot((choices[1:Nt].+1)/2,"k.")
#
# plot(state[1:Nt],"k-")
# plot(pstate[1:Nt],".r--")
#



################ LL vs sigma ###########
# PDwVector=0.05:0.05:0.95
# PBiasVector=0.05:0.05:0.95
#
#
# P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
# ll=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)
#
# SIGMA=0.05:0.01:1
# Ll=zeros(length(SIGMA))
# for isigma in 1:length(SIGMA)
#         x[1]=SIGMA[isigma]
#         P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
#         Ll[isigma]=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)
# end
#
# figure()
# plot(SIGMA,Ll)



############## fitting ############
lower=[0.05,-1.0,-10.0]
upper=[3.0,1.0,10.0]
PossibleOutputs=[1,2]

#### compute loglikelihood Original ####
POriginal=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
LlOriginal=ComputeNegativeLogLikelihood(POriginal,T,choices,PiInitialOriginal)


Nconditions=2
Nstates=2
XInitial=zeros(Nconditions,length(lower))
TInitialAll=zeros(Nconditions,Nstates,Nstates)
ConfideceIntervals=zeros(Nconditions,length(lower)+Nstates)
Ll=zeros(Nconditions)
ParamFit=zeros(Nconditions,length(lower)+Nstates)
PiInitial=zeros(Nconditions,Nstates)


for icondition in 1:Nconditions
    println("icondition:", icondition)
    #random initial conditions
    for iparam in 1:length(lower)
        XInitial[icondition,iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
    end

    pdwdw=rand()
    pbiasbias=rand()
    TInitial=[pdwdw 1-pdwdw ; 1-pbiasbias pbiasbias]
    TInitialAll[icondition,:,:]=TInitial
    aux=rand()
    PiInitial[icondition,1]=aux
    PiInitial[icondition,2]=1-aux

    PNew,TNew,PiNew,Ll[icondition],ParamFit[icondition,:],xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,XInitial[icondition,:],lower,upper,TInitial,PiInitial[icondition,:],PossibleOutputs,consts,y)
    ConfideceIntervals[icondition,:]=ComputeConfidenceIntervals(stim,delays,idelays,choices,past_choices,past_rewards,args,xfit,lower,upper,TNew,PiNew,PossibleOutputs,consts,y)

end


# LL=zeros(length(PDwVector),length(PBiasVector))
# for idw in 1:length(PDwVector)
#     for ibias in 1:length(PBiasVector)
#         LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwVector[idw],PBiasVector[ibias],PrDw,PrBias,choices)
#         #LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PDwVector[idw],PBiasVector[ibias],choices)
#
#     end
# end
# figure()
# imshow(LL,origin="lower",extent=[PBiasVector[1],PBiasVector[end],PDwVector[1],PDwVector[end]],aspect="auto",cmap="hot")
# xlabel("PbiasBias")
# ylabel("PDwDw")
# plot([ PBiasBias],[PDwDw],"bo")
#
# #plot( [ PrBias],[PrDw],"bo")
#
#
# a=findall(x->x==minimum(LL),LL)
# plot([ PBiasVector[a[1][2]]],[PDwVector[a[1][1]]],"bs")
#
# colorbar()
# show()




# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*".jld"
#
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_x0_betal_pdwdw_pbiasbias_Ntrials"*string(Ntrials)*".jld"

save(filename_save,"x",x,"args",args,"y",y,"consts",consts,"XInitial",XInitial,"Ll",Ll,
"PiInitial",PiInitial,"TInitialAll",TInitialAll,"ConfideceIntervals",ConfideceIntervals,
"LlOriginal",LlOriginal)
