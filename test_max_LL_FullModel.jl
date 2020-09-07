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




args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    PDwDw,     PBiasBias]
param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e4)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)


T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]
PiInitial=[0 1]
#
choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
#pr,pstate=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)
#ll=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)

#
# indexDw=findall(x->x==1,state[1:Ntrials])
# indexBias=findall(x->x==0,state[1:Ntrials])
# println(mean((choices[indexDw].+1)./2)," ",mean((choices[indexBias].+1)./2))



# #
# Nt=30
# figure()
#
# plot((choices[1:Nt].+1)/2,"k.")
#
# plot(state[1:Nt],"k-")
# plot(pstate[1:Nt],".r--")
#
# PDwVector=0.05:0.05:0.95
# PBiasVector=0.05:0.05:0.95
#

P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
ll=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)

SIGMA=0.05:0.01:1
Ll=zeros(length(SIGMA))
for isigma in 1:length(SIGMA)
        x[6]=SIGMA[isigma]
        P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
        Ll[isigma]=ComputeNegativeLogLikelihood(P,T,choices,PiInitial)
end

figure()
plot(SIGMA,Ll)
xx=zeros(typeof(SIGMA[1]),1)
xx[1]=0.4
lower=[0.05]
upper=[3.0]

#llmax=MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,T,PiInitial,args,x,xx)
#MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,T,PiInitial,args,x,xx,lower,upper)

PossibleOutputs=[1,2]
tol=1e-3
x2=x[:]
x2[6]=xx[1]
PInitial=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x2)
a=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,x,T,PiInitial,PossibleOutputs,tol,xx,lower,upper)


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



#
# # filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*".jld"
# #
# # filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# # filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*".jld"
# # filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# # filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
# filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_hmm_only_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
#
#
#
# save(filename_save,"x",x,"Ymin",Ymin,"Yini",Yini,"LL",LL,"Hess",Hess,"args",args,"LlOriginal",LlOriginal)
