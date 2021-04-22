using PyPlot
using Statistics
using JLD
path_functions="/home/genis/wm_mice/scripts/functions/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")
#noDrug
filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_WM096_repeating085.jld"
drug="Saline"

#Drug
# filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_WM09_repeating085.jld"
# drug="NR2B"


data=load(filename_save)

rewards=data["rewards"]
choices=data["choices"]
idelays=data["idelays"]
stim=data["stim"]
delays=data["delays"]
state=data["state"]
past_choices=data["past_choices"]
past_rewards=data["past_rewards"]

##############  P correct vs delay ############


Pc_delay=zeros(length(delays))
PcDwDelay=zeros(length(delays))
PcBiasDelay=zeros(length(delays))

for idelay in 1:length(delays)
    println(idelay)
    index=findall(x->x==idelay,idelays)
    Pc_delay[idelay]=mean( rewards[index])
    state2=state[index]
    rewards2=rewards[index]
    index_dw=findall(x->x==1,state2)
    index_bias=findall(x->x==2,state2)

    PcDwDelay[idelay]=mean( rewards2[index_dw])
    PcBiasDelay[idelay]=mean( rewards2[index_bias])


end

fig,axes=subplots(1,1,figsize=(6/2.54,6/2.54))
axes.plot([delays[1],delays[end]],[0.5,0.5],"k--")
axes.plot(delays,Pc_delay,"k.-",label="All trials")
axes.plot(delays,PcDwDelay,".-",color="orange",label="WM module")
axes.plot(delays,PcBiasDelay,".-",color="purple",label="Repeating module")
#
fontsize=9
xticks=[0,500,1000]
yticks=[0.5,0.75,1]

axes.set_xticks(xticks)
axes.set_xticklabels(xticks,fontsize=fontsize)


axes.set_yticks(yticks)
axes.set_yticklabels(yticks,fontsize=fontsize)
axes.spines["right"].set_visible(false)
axes.spines["top"].set_visible(false)


axes.legend(frameon=false,fontsize=fontsize)
axes.set_xlabel("Delay (a.u.)",fontsize=fontsize)
axes.set_ylabel("Accuracy",fontsize=fontsize)

fig.tight_layout()

fig.show()

fig.savefig("/home/genis/wm_mice/figures/caixa_grant_accuracy_modules"*drug*".pdf")



#
# ############### Prob repeat ############
repeat=(choices.*past_choices[:,1].+1)/2.
reward=past_rewards[:,1]
Prep_delay=zeros(length(delays))
PrepDwDelay=zeros(length(delays))
PrepBiasDelay=zeros(length(delays))

PrepBiasDelayCorrect=zeros(length(delays))
PrepBiasDelayError=zeros(length(delays))

for idelay in 1:length(delays)
    println(idelay)
    index=findall(x->x==idelay,idelays)
    Prep_delay[idelay]=mean(repeat[index])

    state2=state[index]
    repeat2=repeat[index]
    reward2=rewards[index]

    index_dw=findall(x->x==1,state2)
    index_bias=findall(x->x==2,state2)
    reward2=reward2[index_bias]
    repeat3=repeat2[index_bias]

    PrepDwDelay[idelay]=mean( repeat2[index_dw])
    PrepBiasDelay[idelay]=mean( repeat2[index_bias])

    indexCorrect=findall(x->x==1,reward2)
    indexError=findall(x->x==0,reward2)

    PrepBiasDelayCorrect[idelay]=mean( repeat3[indexCorrect])
    PrepBiasDelayError[idelay]=mean( repeat3[indexError])

end
#

fig,axes=subplots(1,1,figsize=(6/2.54,6/2.54))
axes.plot([delays[1],delays[end]],[0.5,0.5],"k--")
axes.plot(delays,Prep_delay,"k.-",label="All trials")
axes.plot(delays,PrepDwDelay,".-",color="orange",label="WM module")
axes.plot(delays,PrepBiasDelay,".-",color="purple",label="Repeating module")
#
xticks=[0,500,1000]
yticks=[0.5,0.75,1]

axes.set_xticks(xticks)
axes.set_xticklabels(xticks,fontsize=fontsize)


axes.set_yticks(yticks)
axes.set_yticklabels(yticks,fontsize=fontsize)
axes.spines["right"].set_visible(false)
axes.spines["top"].set_visible(false)


axes.legend(frameon=false,fontsize=fontsize)
axes.set_xlabel("Delay (a.u.)",fontsize=fontsize)
axes.set_ylabel("Probability of repeat",fontsize=fontsize)

fig.tight_layout()

fig.show()

fig.savefig("/home/genis/wm_mice/figures/caixa_grant_repeat_modules"*drug*".pdf")





############### Prob Right| s=Right, choices-1=right choice-1=left ############

Prr=zeros(length(delays))
Prl=zeros(length(delays))

PrBias=zeros(length(delays))
PrDw=zeros(length(delays))

choices_r=(choices.+1)/2
#past_choice_r=(past_choice_r.+1)/2

for idelay in 1:length(delays)
    indexdelay= idelays.==idelay
    index_r_current= stim.==1
    index_r= past_choices[:,1].==1
    index_l=past_choices[:,1].==-1

    index_rd=(indexdelay.*index_r_current).*index_r

    index_ld=(indexdelay.*index_r_current).*index_l

    #println("size rd) ",size(index_rd) )
    Prr[idelay]=mean(choices_r[index_rd])
    Prl[idelay]=mean(choices_r[index_ld])


    # state2=state[index]
    # choices_r2=choices_r[index]
    #
    # index_dw=findall(x->x==1,state2)
    # index_bias=findall(x->x==2,state2)
    #
    # PrBias[idelay]=mean(choices_r2[index_bias])
    # PrDw[idelay]=mean(choices_r2[index_dw])

end

fig,axes=subplots(1,1,figsize=(6/2.54,6/2.54))

axes.plot([delays[1],delays[end]],[0.5,0.5],"k--")
axes.plot(delays,Pc_delay,"k.-",label="All trials")
axes.plot(delays,Prr,".-",color="blue",label=L"$X^+$")
axes.plot(delays,Prl,".-",color="red",label=L"$Y^+$")


xticks=[0,500,1000]
yticks=[0.5,0.75,1]

axes.set_xticks(xticks)
axes.set_xticklabels(xticks,fontsize=fontsize)


axes.set_yticks(yticks)
axes.set_yticklabels(yticks,fontsize=fontsize)
axes.spines["right"].set_visible(false)
axes.spines["top"].set_visible(false)


axes.legend(frameon=false,fontsize=fontsize)
axes.set_xlabel("Delay (a.u.)",fontsize=fontsize)
axes.set_ylabel("Accuracy",fontsize=fontsize)

fig.tight_layout()

fig.show()

fig.savefig("/home/genis/wm_mice/figures/caixa_grant_accuracy_history"*drug*".pdf")








# figure()
# PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")
#
# PyPlot.plot(delays,Prep_delay,"ko-",label="All trials")
# PyPlot.plot(delays,PrepDwDelay,"o-",color="Orange",label="WM module")
# PyPlot.plot(delays,PrepBiasDelay,"o-",color="Purple",label="Repeating module")
# legend()
# xlabel("Delay")
# ylabel("Probability of repeat")
# show()
# savefig(path_figures*"Prepeat_delay.pdf")
#
#
# figure()
# PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")
#
# PyPlot.plot(delays,PrepBiasDelayCorrect,"o-",label="After Correct")
# PyPlot.plot(delays,PrepBiasDelayError,"o-",label="After Error")
# legend()
# xlabel("Delay")
# ylabel("Probability of repeat")
# savefig(path_figures*"Prepeat_AfterCorrect-Error_delay.png")
#
# show()
#
#
# ############### Prob Right ############
#
# Pr=zeros(length(delays))
# PrBias=zeros(length(delays))
# PrDw=zeros(length(delays))
#
# choices_r=(choices.+1)/2
#
# for idelay in 1:length(delays)
#     index=findall(x->x==idelay,idelays)
#     Pr[idelay]=mean(choices_r[index])
#
#     state2=state[index]
#     choices_r2=choices_r[index]
#
#     index_dw=findall(x->x==1,state2)
#     index_bias=findall(x->x==2,state2)
#
#     PrBias[idelay]=mean(choices_r2[index_bias])
#     PrDw[idelay]=mean(choices_r2[index_dw])
#
# end
#
#
# figure()
# PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")
#
# PyPlot.plot(delays,Pr,"o-",label="all trials")
# #
# PyPlot.plot(delays,PrDw,"o-",label="Dw module")
# PyPlot.plot(delays,PrBias,"o-",label="Bias module")
# legend()
# xlabel("Delay")
# ylabel("P(d=R)")
# savefig(path_figures*"PR_delay.png")
#
# show()
#
#
#
#
#
#
#
#
#
#
#

#
#
#
#
#
#
# # PyPlot.plot(delays,PrDw,"o-",label="Dw module")
# # PyPlot.plot(delays,PrBias,"o-",label="Bias module")
# legend()
# xlabel("Delay")
# #ylabel("P(d(t)=R|S(t)=R,d(t-1)=X)")
# ylabel("Accuracy")
#
# savefig(path_figures*"PR_givenR_or_L_delay.png")
#
# show()
#
# filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_WM07_repeating05.jld"
#
# JLD.save(filename_save,"delays",delays,"args",args,"y",y,"consts",consts,"Prep_delay"
# ,Prep_delay,"PrepDwDelay",PrepDwDelay,"PrepBiasDelay",PrepBiasDelay,"Prr",
# Prr,"Prl",Prl,"Pc_delay",Pc_delay,"PcDwDelay",PcDwDelay,"PcBiasDelay",
# PcBiasDelay,"choices",choices,"rewards",rewards,"state",state,"stim",stim)
#
#
#
#
#
#
# # using Pandas
# # PAST_CHOICES=[]
# # PAST_REWARDS=[]
# # for itrial in 1:Ntrials
# #     push!(PAST_CHOICES,past_choices[itrial,:])
# #     push!(PAST_REWARDS,past_rewards[itrial,:])
# # end
# # dict=Dict(:choices=>choices,:stim=>stim,:past_choices=>PAST_CHOICES,:past_rewards=>PAST_REWARDS,:idelays=>idelays)
# # df=Pandas.DataFrame(dict)
# # #
# # filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_WM09_repeating05.json"
# # Pandas.to_json(df,filename_save)
