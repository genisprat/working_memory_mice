using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches
using LinearAlgebra
# path_functions="/home/genis/wm_mice/scripts/functions/"
# path_figures="/home/genis/wm_mice/figures/"
#
# include(path_functions*"functions_wm_mice.jl")
# include(path_functions*"function_simulations.jl")

#
# save(path_results*filename_save,"x",x,"args",args,"y",y,"consts",consts,"XInitial",XInitial,"Ll",Ll,
# "PiInitial",PiInitial,"TInitialAll",TInitialAll,"ConfideceIntervals",ConfideceIntervals,
# "LlOriginal",LlOriginal,"T",T,"ParamFit",ParamFit)
#


#args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
#x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.9,     0.8]
#param=make_dict2(args,x)
#delays=[0.0,100,200,300,500,800,1000]
#Ntrials=Int(100)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

#figure()
#iOriginalParam=[7,8] #beta_w beta_l

#iOriginalParam=[2,6] #c2 sigma

NTrials=[100,10000]
NDataSets=[1000,100]
fig,ax= subplots(1,length(NTrials),figsize=(10,3))
for itrial in 1:length(NTrials)
    path="/home/genis/wm_mice/results/"
    filename_save=path*"minimize_sigma_x0_betal_pdwdw_pbiasbias_NDataSet"*string(NDataSets[itrial])*"Ntrials"*string(NTrials[itrial])*".jld"


    data=load(filename_save)
    CI=data["ConfideceIntervals"].s
    T=data["T"]
    iDataSet=1
    index=findall(x->x!=-1,CI[iDataSet,:,1]) ##Only values with a minimum


    #p_original=[data["x"][2],data["x"][6]]
    p_original=vcat([T[1,1]], [T[2,2]] ,data["x"])


    #index=findall(!isnan,data["LL"][iDataSet,:])
    Ll=data["Ll"].s[iDataSet,index]
    imin=findall(x->x==minimum(Ll),Ll)[1]
    println(imin," ",itrial)

    ParamFitPlot=data["ParamFit"].s[iDataSet,index,:]
    Ymin=ParamFitPlot[imin,:]

    err=CI[iDataSet,imin,:]
    println("Ymin: ", Ymin)
    for i in 1:length(Ymin)
        x=0.5*rand(length(index)).+i.-0.25
        ax[itrial].plot(ParamFitPlot[:,i],x,"k.")
        ax[itrial].errorbar([Ymin[i]],[i],xerr=[err[i]],fmt="b.")
        ax[itrial].plot([p_original[i]],[i],"r.")
        ax[itrial].set_title("Ntrials"*string(NTrials[itrial]))

    end
    i=1
    ax[itrial].errorbar([Ymin[i]],[i],xerr=[err[i]],fmt="b.",label="Param Best")
    ax[itrial].plot([p_original[i]],[i],"r.",label="Param Original")
    ax[itrial].set_yticks([1,2,3,4,5])
    ax[itrial].set_yticklabels(["pWmWm","pBiasBias","sigma","x0","BetaL"])


end
ax[1].legend(loc=4)
fig.tight_layout()
#
fig,ax2= subplots(1,length(NTrials),figsize=(10,3))
Nparam=5
fig3,ax3= subplots(Nparam,length(NTrials),figsize=(10,10))
NDataSets2=100
param_min=zeros(length(NTrials),NDataSets2,Nparam)
BINS=[]

for itrial in 1:length(NTrials)
    LlMinDataSet=zeros(NDataSets[itrial])

    path="/home/genis/wm_mice/results/"
    filename_save=path*"minimize_sigma_x0_betal_pdwdw_pbiasbias_NDataSet"*string(NDataSets[itrial])*"Ntrials"*string(NTrials[itrial])*".jld"
    data=load(filename_save)


    LlOriginal=data["LlOriginal"].s
    Ll=data["Ll"].s

    for iDataSet in 1:NDataSets2


        CI=data["ConfideceIntervals"].s
        T=data["T"]
        index=findall(x->x!=-1,CI[iDataSet,:,1]) ##Only values with a minimum


        ll_aux=Ll[iDataSet,index]
        imin=findall(x->x==minimum(ll_aux),ll_aux)
        if length(imin)>1
            imin=imin[1]
        end
        println("iDataSet",iDataSet," imin: ",imin[1]," len ll_aux",length(ll_aux))

        LlMinDataSet[iDataSet]=ll_aux[imin[1]]
        ParamFitPlot=data["ParamFit"].s[iDataSet,index,:]
        Ymin=ParamFitPlot[imin[1],:]

        for iparam in 1:Nparam
            println("a, ",Ymin[iparam]," ")
            param_min[itrial,iDataSet,iparam]=Ymin[iparam]
        end


    end

    ax2[itrial].set_title("Ntrials"*string(NTrials[itrial]))
    if NDataSets[itrial]<20
        ax2[itrial].plot(LlOriginal,label="LLOriginal","r.")
        ax2[itrial].plot(LlMinDataSet,label="min_dataSet","k.")
    else
        ax2[itrial].plot(LlOriginal[1:20],label="LLOriginal","r.")
        ax2[itrial].plot(LlMinDataSet[1:20],label="min_dataSet","k.")
    end

end
ax2[1].legend()

fig.tight_layout()

#bmin=[minimum(param_min[:,:,1]),minimum(param_min[:,:,2])]
#bmax=[maximum(param_min[:,:,1]),maximum(param_min[:,:,2])]
for itrial in 1:length(NTrials)
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_muk_betaw_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_betaw_betal_only_history_bias_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"
    #filename_save="/home/genis/wm_mice/synthetic_data/minimize_sigma_c2_wm_only_Ntrials"*string(NTRIALS[itrial])*"_NDataSets"*string(NDataSets[itrial])*".jld"

    path="/home/genis/wm_mice/results/"
    filename_save=path*"minimize_sigma_x0_betal_pdwdw_pbiasbias_NDataSet"*string(NDataSets[itrial])*"Ntrials"*string(NTrials[itrial])*".jld"

    data=load(filename_save)
    x=data["x"]

    T=data["T"]
    p_original=vcat([T[1,1]], [T[2,2]] ,data["x"])


    ax3[1,itrial].set_title("Ntrials"*string(NTrials[itrial]))
    Nbins=[1000 1000 1000 1000 1000]
    aux_xlim=[[0, 1],[0, 1],[0, 3],[-1,1],[-10, 10]]
    println(aux_xlim[1,:])
    ylabels=["pWmWm","pBiasBias","sigma","x0","BetaL"]
    for iparam in 1:Nparam
        #ax3[iparam,itrial].hist(param_min[itrial,:,iparam],range=(minimum(param_min[itrial,:,iparam]),maximum(param_min[itrial,:,iparam])),bins=100)
        ax3[6-iparam,itrial].hist(param_min[itrial,:,iparam],range=(-10,10),bins=Nbins[iparam])

        ax3[6-iparam,itrial].plot([p_original[iparam],p_original[iparam]],[0,5],"r-")
        ax3[6-iparam,itrial].set_xlim([aux_xlim[iparam][1],aux_xlim[iparam][2]])

        ax3[6-iparam,1].set_ylabel(ylabels[iparam])
    end

end


fig3.tight_layout()
show()
#
