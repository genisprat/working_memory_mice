using PyPlot
using Statistics
path_functions="/home/genis/wm_mice/scripts/functions/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")




consts=["mu_k","c2","c4","mu_b","beta_w","tau_w","tau_l", "mu0_a","tau_a"]
y=[    0.3,  1.2, 1.0, -0.00,     3.0,     10,     10,  0.1,3]

args=["sigma","x0","beta_l"]
x=[0.3,-0.1,-1.0]

PDwDw=0.9
PBiasBias=0.1

T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]



param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e6)
choices,rewards,state,stim,past_choices,past_rewards,idelays,BiasAttraction=create_data(Ntrials,delays,T,args,x,consts,y)
#change stim from 1,2 to -1,1
#a=findall(x->x==1,stim)
#b=findall(x->x==-1,stim)
# stim[a].=-1
# stim[b].=1

############### state transitions ##############


figure()
PyPlot.plot(state[1:300])
xlabel("Trial number")
ylabel("State")
show()
savefig(path_figures*"state_time.png")

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

figure()
PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")
PyPlot.plot(delays,Pc_delay,"o-",label="All trials")
PyPlot.plot(delays,PcDwDelay,"o-",label="Dw module")
PyPlot.plot(delays,PcBiasDelay,"o-",label="Bias module")

legend()
xlabel("Delay")
ylabel("Accuracy")
show()
savefig(path_figures*"Accuracy_delay.png")

############### Prob repeat ############
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


figure()
PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")

PyPlot.plot(delays,Prep_delay,"o-",label="All trials")
PyPlot.plot(delays,PrepDwDelay,"o-",label="Dw module")
PyPlot.plot(delays,PrepBiasDelay,"o-",label="Bias module")
legend()
xlabel("Delay")
ylabel("Probability of repeat")
show()
savefig(path_figures*"Prepeat_delay.png")


figure()
PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")

PyPlot.plot(delays,PrepBiasDelayCorrect,"o-",label="After Correct")
PyPlot.plot(delays,PrepBiasDelayError,"o-",label="After Error")
legend()
xlabel("Delay")
ylabel("Probability of repeat")
savefig(path_figures*"Prepeat_AfterCorrect-Error_delay.png")

show()


############### Prob Right ############

Pr=zeros(length(delays))
PrBias=zeros(length(delays))
PrDw=zeros(length(delays))

choices_r=(choices.+1)/2

for idelay in 1:length(delays)
    index=findall(x->x==idelay,idelays)
    Pr[idelay]=mean(choices_r[index])

    state2=state[index]
    choices_r2=choices_r[index]

    index_dw=findall(x->x==1,state2)
    index_bias=findall(x->x==2,state2)

    PrBias[idelay]=mean(choices_r2[index_bias])
    PrDw[idelay]=mean(choices_r2[index_dw])

end


figure()
PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")

PyPlot.plot(delays,Pr,"o-",label="all trials")
#
PyPlot.plot(delays,PrDw,"o-",label="Dw module")
PyPlot.plot(delays,PrBias,"o-",label="Bias module")
legend()
xlabel("Delay")
ylabel("P(d=R)")
savefig(path_figures*"PR_delay.png")

show()







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




figure()
PyPlot.plot([delays[1],delays[end]],[0.5,0.5],"k--")

PyPlot.plot(delays,Prr,"o-",label="X=R")
PyPlot.plot(delays,Prl,"o-",label="X=L")
#
# PyPlot.plot(delays,PrDw,"o-",label="Dw module")
# PyPlot.plot(delays,PrBias,"o-",label="Bias module")
legend()
xlabel("Delay")
ylabel("P(d(t)=R|S(t)=R,d(t-1)=X)")
savefig(path_figures*"PR_givenR_or_L_delay.png")

show()


# using Pandas
# PAST_CHOICES=[]
# PAST_REWARDS=[]
# for itrial in 1:Ntrials
#     push!(PAST_CHOICES,past_choices[itrial,:])
#     push!(PAST_REWARDS,past_rewards[itrial,:])
# end
# dict=Dict(:choices=>choices,:stim=>stim,:past_choices=>PAST_CHOICES,:past_rewards=>PAST_REWARDS,:idelays=>idelays)
# df=Pandas.DataFrame(dict)
# #
# filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_attraction_bias.json"
# Pandas.to_json(df,filename_save)
