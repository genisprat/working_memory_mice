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
Ntrials=Int(1e4)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
PDwDw=0.9
PBiasBias=0.1
PrDw=0.9
PrBias=0.3

#
choice,state=create_data_hmm(PDwDw,PBiasBias,PrDw,PrBias,Ntrials)
#pr,pstate=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choice)
#ll=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choice)

#
# indexDw=findall(x->x==1,state[1:Ntrials])
# indexBias=findall(x->x==0,state[1:Ntrials])
# println(mean((choice[indexDw].+1)./2)," ",mean((choice[indexBias].+1)./2))



# #
# Nt=30
# figure()
#
# plot((choice[1:Nt].+1)/2,"k.")
#
# plot(state[1:Nt],"k-")
# plot(pstate[1:Nt],".r--")
#
# PDwVector=0.05:0.05:0.95
# PBiasVector=0.05:0.05:0.95
#
LL=zeros(length(PDwVector),length(PBiasVector))
for idw in 1:length(PDwVector)
    for ibias in 1:length(PBiasVector)
        LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwVector[idw],PBiasVector[ibias],PrDw,PrBias,choice)
        #LL[idw,ibias]=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PDwVector[idw],PBiasVector[ibias],choice)

    end
end
figure()
imshow(LL,origin="lower",extent=[PBiasVector[1],PBiasVector[end],PDwVector[1],PDwVector[end]],aspect="auto",cmap="hot")
xlabel("PbiasBias")
ylabel("PDwDw")
plot([ PBiasBias],[PDwDw],"bo")

#plot( [ PrBias],[PrDw],"bo")


a=findall(x->x==minimum(LL),LL)
plot([ PBiasVector[a[1][2]]],[PDwVector[a[1][1]]],"bs")

colorbar()
show()




# function LL_f(y)
#     #println("hola")
#     z=zeros(typeof(y[1]),length(x))
#     z[:]=x[:]
#     ### sigma c2
#     #z[2]=y[1]
#     #z[6]=y[2]
#     ### mu_k beta_w
#     z[2]=y[1]
#     z[6]=y[2]
#     #println("vamos")
#     return Compute_negative_LL_WM_module(stim,delays,idelays,choices,args,x)
# end
#
#y=[x[1],x[7]]


# lower=[-10.0,-10.0]
# upper=[10.0,10.0]
# Nconditions=100
# Ymin=zeros(Nconditions,length(lower))
# Yini=zeros(Nconditions,length(lower))
# LL=zeros(Nconditions)
# Hess=zeros(Nconditions,length(lower),length(lower))
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




#
# NDataSets=10
# Nconditions=10
# lower=[0.,0.0]
# upper=[1.0,1.0]
#
# Ymin=zeros(NDataSets,Nconditions,length(lower))
# Yini=zeros(NDataSets,Nconditions,length(lower))
# LL=zeros(NDataSets,Nconditions)
# Hess=zeros(NDataSets,Nconditions,length(lower),length(lower))
# LlOriginal=zeros(NDataSets)
#
# for iDataSet in 1:NDataSets
#     println(iDataSet)
#
#     choice,state=create_data_hmm(PDwDw,PBiasBias,PrDw,PrBias,Ntrials)
#
#     LlOriginal[iDataSet]=Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choice)
#
#     function LL_f2(y)
#         return Compute_negative_LL_hmm_module(y[1],y[2],PrDw,PrBias,choice)
#     end
#
#
#
#     for icondition in 1:Nconditions
#         #choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
#         aux=rand(length(upper))
#         y=aux.*(upper-lower).+lower
#
#         res=optimize(LL_f2,lower,upper, y, Fminbox(LBFGS(linesearch = BackTracking(order=2))); autodiff = :forward)
#         Ymin[iDataSet,icondition,:]=res.minimizer
#         Yini[iDataSet,icondition,:]=res.initial_x
#         Hess[iDataSet,icondition,:,:]=ForwardDiff.hessian(LL_f2,res.minimizer)
#         LL[iDataSet,icondition]=res.minimum
#
#     end
# end
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
