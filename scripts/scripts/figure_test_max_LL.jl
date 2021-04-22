using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
using LinearAlgebra
path_functions="/home/genis/wm_mice/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")

args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.9,     0.8]
#param=make_dict2(args,x)
#delays=[0.0,100,200,300,500,800,1000]
#Ntrials=Int(100)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

#figure()
#iOriginalParam=[7,8] #beta_w beta_l

iOriginalParam=[2,6] #c2 sigma

NTRIALS=[10,100,1000,10000]
NDataSets=[100,100,100,100]

fig,ax= subplots(1,length(NTRIALS),figsize=(10,3))
for itrial in 1:length(NTRIALS)
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(NTRIALS[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_muk_betaw_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"

    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"


    data=load(filename_save)
    iDataSet=1
    #p_original=[data["x"][2],data["x"][6]]
    p_original=[data["x"][iOriginalParam[1]],data["x"][iOriginalParam[2]]]


    index=findall(!isnan,data["LL"][iDataSet,:])
    LL=data["LL"][iDataSet,index]
    imin=findall(x->x==minimum(LL),LL)[1]
    println(imin," ",itrial)
    HInv=inv(data["Hess"][iDataSet,imin,:,:])
    auxHes=diag(HInv)
    if all(y->y>0,auxHes)
        err=2*sqrt.(auxHes) #confidence interval
    else
        err=NaN*ones(length(auxHes))
    end
    Ymin=data["Ymin"][iDataSet,imin,:]

    println("Ymin: ", Ymin)
    for i in 1:length(Ymin)
        x=0.5*rand(length(data["Ymin"][iDataSet,:,i])).+i.-0.25
        ax[itrial].plot(data["Ymin"][iDataSet,:,i],x,"k.")
        ax[itrial].errorbar([Ymin[i]],[i],xerr=[err[i]],fmt="k")
        ax[itrial].plot([p_original[i]],[i],"r.")
        ax[itrial].set_title("Ntrials"*string(NTRIALS[itrial]))

    end


end


fig,ax2= subplots(1,length(NTRIALS),figsize=(10,3))

fig3,ax3= subplots(2,length(NTRIALS),figsize=(10,3))
param_min=zeros(length(NTRIALS),NDataSets[1],2)
BINS=[]

for itrial in 1:length(NTRIALS)
    LlMinDataSet=zeros(NDataSets[itrial])

    filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_muk_betaw_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"

    data=load(filename_save)

    LlOriginal=data["LlOriginal"]
    LL=data["LL"]

    for iDataSet in 1:NDataSets[itrial]
        println("iDataSet",iDataSet," ",itrial)

        index=findall(!isnan,LL[iDataSet,:])
        ymin_aux=data["Ymin"][iDataSet,index,:]
        ll_aux=LL[iDataSet,index]
        imin=findall(x->x==minimum(ll_aux),ll_aux)
        if length(imin)>1
            imin=imin[1]
        end
        println("imin: ",imin)

        LlMinDataSet[iDataSet]=ll_aux[imin[1]]
        for iparam in 1:length(param_min[itrial,1,:])
            println("a, ",ymin_aux[imin[1],iparam]," ")
            param_min[itrial,iDataSet,iparam]=ymin_aux[imin[1],iparam]
        end


    end

    ax2[itrial].set_title("Ntrials"*string(NTRIALS[itrial]))
    if NDataSets[itrial]<20
        ax2[itrial].plot(LlOriginal,label="LLOriginal","k.")
        ax2[itrial].plot(LlMinDataSet,label="min_dataSet","r.")
    else
        ax2[itrial].plot(LlOriginal[1:20],label="LLOriginal","k.")
        ax2[itrial].plot(LlMinDataSet[1:20],label="min_dataSet","r.")
    end

end

bmin=[minimum(param_min[:,:,1]),minimum(param_min[:,:,2])]
bmax=[maximum(param_min[:,:,1]),maximum(param_min[:,:,2])]
for itrial in 1:length(NTRIALS)
    filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_muk_betaw_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"

    data=load(filename_save)
    x=data["x"]
    ax3[1,itrial].set_title("Ntrials"*string(NTRIALS[itrial]))

    ax3[1,itrial].hist(param_min[itrial,:,1],range=(bmin[1],bmax[1]),bins=100)
    ax3[2,itrial].hist(param_min[itrial,:,2],range=(bmin[2],bmax[2]),bins=100)
    ax3[1,itrial].plot([x[iOriginalParam[1]],x[iOriginalParam[1]]],[0,5],"k-")
    ax3[2,itrial].plot([x[iOriginalParam[2]],x[iOriginalParam[2]]],[0,5],"k-")
end
ax2[length(NTRIALS)].legend()

show()






#
# function LL_f(y)
#     #println("hola")
#     z=zeros(typeof(y[1]),length(x))
#     z[:]=x[:]
#     z[2]=y[1]
#     z[6]=y[2]
#     #println("vamos")
#     return Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,z)
# end
#
# y=[1.2,0.3]
#
# lower=[0.2,0.05]
# upper=[4,0.6]
#
# Nconditions=100
#
# Ymin=zeros(Nconditions,length(lower))
# Yini=zeros(Nconditions,length(lower))
# LL=zeros(Nconditions)
# Hess=zeros(Nconditions,length(y),length(y))
# for icondition in 1:Nconditions
#     println(icondition)
#     aux=rand(length(upper))
#     y=aux.*(upper-lower).+lower
#
#     res=optimize(LL_f,lower,upper, y, Fminbox(LBFGS(linesearch = BackTracking(order=2))); autodiff = :forward)
#     Ymin[icondition,:]=res.minimizer
#     Yini[icondition,:]=res.initial_x
#     Hess[icondition,:,:]=ForwardDiff.hessian(LL_f,res.minimizer)
#     LL[icondition]=res.minimum
#
# end
#
# save(filename_save,"x",x,"Ymin",Ymin,"Yini",Yini,"LL",LL,"Hess",Hess,"args",args)
