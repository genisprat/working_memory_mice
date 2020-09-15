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

args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.9,     0.8]
param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e3)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

choices,past_choices,past_rewards=create_data_history_bias(Ntrials,args,x)

function LL_f(y)
    #println("hola")
    z=zeros(typeof(y[1]),length(x))
    z[:]=x[:]
    ### sigma c2
    #z[2]=y[1]
    #z[6]=y[2]
    ### mu_k beta_w
    z[7]=y[1]
    z[8]=y[2]
    #println("vamos")
    return Compute_negative_LL_history_bias(choices,past_choices,past_rewards,args,z)
end

y=[x[1],x[7]]
LlOriginal=Compute_negative_LL_history_bias(choices,past_choices,past_rewards,args,x)



# lower=[-10.0,-10.0]
# upper=[10.0,10.0]
# Nconditions=100
# Ymin=zeros(Nconditions,length(lower))
# Yini=zeros(Nconditions,length(lower))
# LL=zeros(Nconditions)
# Hess=zeros(Nconditions,length(y),length(y))
#
#
# for icondition in 1:Nconditions
#     #choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
#     aux=rand(length(upper))
#     y=aux.*(upper-lower).+lower
#
#     res=optimize(LL_f,lower,upper, y, Fminbox(LBFGS(linesearch = BackTracking(order=2))); autodiff = :forward)
#     Ymin[icondition,:]=res.minimizer
#     Yini[icondition,:]=res.initial_x
#     Hess[icondition,:,:]=ForwardDiff.hessian(LL_f,res.minimizer)
#     LL[icondition]=res.minimum
# end


#muk betw



NDataSets=100
Nconditions=10
lower=[-10.0,-10.0]
upper=[10.0,10.0]

Ymin=zeros(NDataSets,Nconditions,length(lower))
Yini=zeros(NDataSets,Nconditions,length(lower))
LL=zeros(NDataSets,Nconditions)
Hess=zeros(NDataSets,Nconditions,length(y),length(y))
LlOriginal=zeros(NDataSets)

for iDataSet in 1:NDataSets
    println(iDataSet)

    choices,past_choices,past_rewards=create_data_history_bias(Ntrials,args,x)

    LlOriginal[iDataSet]=Compute_negative_LL_history_bias(choices,past_choices,past_rewards,args,x)



    function LL_f2(y)
        #println("hola")
        z=zeros(typeof(y[1]),length(x))
        z[:]=x[:]
        ### sigma c2
        #z[2]=y[1]
        #z[6]=y[2]
        ### mu_k beta_w
        z[7]=y[1]
        z[8]=y[2]
        #println("vamos")
        return Compute_negative_LL_history_bias(choices,past_choices,past_rewards,args,z)
    end



    for icondition in 1:Nconditions
        #choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
        aux=rand(length(upper))
        y=aux.*(upper-lower).+lower

        res=optimize(LL_f2,lower,upper, y, Fminbox(LBFGS(linesearch = BackTracking(order=2))); autodiff = :forward)
        Ymin[iDataSet,icondition,:]=res.minimizer
        Yini[iDataSet,icondition,:]=res.initial_x
        Hess[iDataSet,icondition,:,:]=ForwardDiff.hessian(LL_f2,res.minimizer)
        LL[iDataSet,icondition]=res.minimum

    end
end
#
#filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*".jld"

#filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"
#filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*".jld"
filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(Ntrials)*"_NDataSets"*string(NDataSets)*".jld"



save(filename_save,"x",x,"Ymin",Ymin,"Yini",Yini,"LL",LL,"Hess",Hess,"args",args,"LlOriginal",LlOriginal)
