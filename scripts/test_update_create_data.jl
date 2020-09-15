using PyPlot
using Statistics

path_functions="/home/genis/wm_mice/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")

args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.95,     0.8]
param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e6)
choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
#change stim from 1,2 to -1,1
a=findall(x->x==1,stim)
b=findall(x->x==2,stim)
stim[a].=-1
stim[b].=1


############### state transitions ##############


figure()
plot(state[1:300])
xlabel("Trial number")
ylabel("State")
show()
savefig(path_figures*"state_time.png")

##############  P correct vs delay ############
correct=( stim.*choices.+1)./2
Pc_delay=zeros(length(delays))
PcDwDelay=zeros(length(delays))
PcBiasDelay=zeros(length(delays))

for idelay in 1:length(delays)
    println(idelay)
    index=findall(x->x==idelay,idelays)
    Pc_delay[idelay]=mean( correct[index])
    state2=state[index]
    correct2=correct[index]
    index_dw=findall(x->x==1,state2)
    index_bias=findall(x->x==2,state2)

    PcDwDelay[idelay]=mean( correct2[index_dw])
    PcBiasDelay[idelay]=mean( correct2[index_bias])


end

figure()
plot([delays[1],delays[end]],[0.5,0.5],"k--")
plot(delays,Pc_delay,"o-",label="All trials")
plot(delays,PcDwDelay,"o-",label="Dw module")
plot(delays,PcBiasDelay,"o-",label="Bias module")

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
    reward2=reward[index]

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
plot([delays[1],delays[end]],[0.5,0.5],"k--")

plot(delays,Prep_delay,"o-",label="All trials")
plot(delays,PrepDwDelay,"o-",label="Dw module")
plot(delays,PrepBiasDelay,"o-",label="Bias module")
legend()
xlabel("Delay")
ylabel("Probability of repeat")
show()
savefig(path_figures*"Prepeat_delay.png")


figure()
plot([delays[1],delays[end]],[0.5,0.5],"k--")

plot(delays,PrepBiasDelayCorrect,"o-",label="After Correct")
plot(delays,PrepBiasDelayError,"o-",label="After Error")
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
plot([delays[1],delays[end]],[0.5,0.5],"k--")

plot(delays,Pr,"o-",label="All trials")
plot(delays,PrDw,"o-",label="Dw module")
plot(delays,PrBias,"o-",label="Bias module")
legend()
xlabel("Delay")
ylabel("Probability of Right")
savefig(path_figures*"Pright_delay.png")

show()
