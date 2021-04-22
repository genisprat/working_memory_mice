using PyPlot
using Statistics
using JLD
using Distributed

path_functions="/home/genis/wm_mice/scripts/functions/"
path_figures="/home/genis/wm_mice/figures/"


include(path_functions*"functions_wm_mice.jl")
include(path_functions*"functions_mle.jl")

#include(path_functions*"function_simulations.jl")



#
# consts=["mu_k","c2","c4","mu_b","beta_w","tau_w","tau_l", "mu0_a","tau_a"]
# y=[    0.3,  1.2, 1.0, -0.00,     3.0,     10,     10,  0.1,3]
#
# args=["sigma","x0","beta_l"]
# x=[0.3,-0.1,-1.0]

# Repeating bias
consts=["mu_k","c2","c4","mu_b","beta_r","tau_r","mu0_a","tau_a"]
y=[      0.3,  1.2, 1.0, 0.00,     1.,     2,     0.0,     1]

args=["sigma","x0"]
x=[0.3,0.0]

#
PDwDw=0.96
PBiasBias=0.85


# PDwDw=0.7
# PBiasBias=0.5


T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]



param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e4)
choices,rewards,state,stim,past_choices,past_rewards,idelays,BiasAttraction=create_data(Ntrials,delays,T,args,x,consts,y)

stim_index=Int.( ((stim.+1)/2).+1)
choices_index=Int.( ((choices.+1)/2).+1)
P=ComputeEmissionProb(stim_index,delays,idelays,choices_index,past_choices,past_rewards,args,x,consts,y)
InitialP=[1,0]
Piinitial,forward,backward,gamma,xi,ll=ProbabilityState(P,T,choices_index,InitialP)

trial_i=400
trial_f=600
# plot(forward[100:200,1],"y-")
# plot(backward[100:200,1],"r-")

fontsize=8
fig,axes=subplots(1,1,figsize=(12/2.54,6/2.54))
#axes.plot([delays[1],delays[end]],[0.5,0.5],"k--")
axes.plot(state[trial_i:trial_f].-1,"k-",label="State")
axes.plot(gamma[trial_i:trial_f,1],"-",color="grey",label="Estimated Probability")
xticks=[0,Int((trial_f-trial_i)/2), Int( trial_f-trial_i)]
yticks=[0,1]

axes.set_xticks(xticks)
axes.set_xticklabels(xticks,fontsize=fontsize)


axes.set_yticks(yticks)
axes.set_yticklabels(yticks,fontsize=fontsize)
axes.spines["right"].set_visible(false)
axes.spines["top"].set_visible(false)


axes.legend(frameon=false,fontsize=fontsize)
axes.set_xlabel("Trial number",fontsize=fontsize)
axes.set_ylabel("",fontsize=fontsize)

fig.tight_layout()

fig.show()
fig.savefig("/home/genis/wm_mice/figures/caixa_grant_prob_estimated.pdf")
