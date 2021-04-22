#using PyPlot
using Distributed
using SharedArrays

# @everywhere using Statistics
# @everywhere using Optim
# @everywhere using ForwardDiff
using JLD
#
rmprocs(workers())
addprocs(20)

#cluster
@everywhere path_functions="/home/hcli64/hcli64751/wm_mice/scripts/functions/"
@everywhere path_results="/home/hcli64/hcli64751/wm_mice/results/"

#local
# @everywhere path_functions="/home/genis/wm_mice/scripts/functions/"
# @everywhere path_results="/home/genis/wm_mice/results/"


@everywhere include(path_functions*"functions_wm_mice.jl")
@everywhere include(path_functions*"functions_mle.jl")




PDwDw=0.9
PBiasBias=0.1
PrDw=0.9
PrBias=0.3




consts=["tau_w","tau_l"]
y=[   10,     10]

args=["mu_k","c2","c4","mu_b","sigma","x0","beta_l","beta_w"]
x=[   0.3,  1.2,  1.0, -0.05,   0.3,  0.15,  -1.0,    3.0]

lower=[0,    0.1,  0.1,  -3, 0.05, -3, -10.0, -10]
upper=[10,   10,   10.0, 3.0,  10.0, 3,    10.0,  10]


#
# consts=["c4","mu_b","x0","beta_l","beta_w","tau_w","tau_l"]
# y=[    1.0, -0.05,  0.15,  -1.0,    3.0,     10,     10]
#
# args=["mu_k","sigma","c2"]
# x=[   0.3,  0.3, 1.2  ]
#
# lower=[0,    0.1,0.1]
# upper=[10,   10,10]


PossibleOutputs=[1,2]

#XInitial[7.830735073501732, 4.521874327905985, 7.39729817002145, -2.425982781120749, 0.4757672169960357, 1.3863976946270498, 0.9515447578404501, 9.790088672546968]

param=make_dict(args,x,consts,y)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e4)
#choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)


T=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]
PiInitialOriginal=[0 1]
#

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



NDataSets=2
Nconditions=100
Nstates=2
XInitial=SharedArray{Float64}(NDataSets,Nconditions,length(lower))#zeros(Nconditions,length(lower))
TInitialAll=SharedArray{Float64}(NDataSets,Nconditions,Nstates,Nstates)#zeros(Nconditions,Nstates,Nstates)
ConfideceIntervals=SharedArray{Float64}(NDataSets,Nconditions,length(lower)+Nstates)#zeros(Nconditions,length(lower)+Nstates)
Ll=SharedArray{Float64}(NDataSets,Nconditions)#zeros(Nconditions)
ParamFit=SharedArray{Float64}(NDataSets,Nconditions,length(lower)+Nstates)#zeros(Nconditions,length(lower)+Nstates)
PiInitial=SharedArray{Float64}(NDataSets,Nconditions,Nstates)#zeros(Nconditions,Nstates)
LlOriginal=SharedArray{Float64}(NDataSets)
println("puta joder")
iDataSet=1
for iDataSet in 1:NDataSets
#    println("iDATASET:", iDataSet)
    ### create_data_set ###########
    choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,T,args,x,consts,y)
    #### compute loglikelihood Original ####

    POriginal=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts,y)
    LlOriginal[iDataSet]=ComputeNegativeLogLikelihood(POriginal,T,choices,PiInitialOriginal)

    @sync @distributed for icondition in 1:Nconditions
    #for icondition in 1:Nconditions

        println("icondition:", icondition)
        #random initial conditions
        for iparam in 1:length(lower)
            #random Initial Conditions
            XInitial[iDataSet,icondition,iparam]=lower[iparam]+ (upper[iparam]-lower[iparam])*rand()
            #True Param
            #XInitial[iDataSet,icondition,iparam]=x[iparam]
        end
        #
        pdwdw=rand()
        pbiasbias=rand()


        #Random Inital conditions
        TInitial=[pdwdw 1-pdwdw ; 1-pbiasbias pbiasbias]
        #True Param
        #TInitial=[PDwDw 1-PDwDw; 1-PBiasBias PBiasBias]

        #println(TInitial)
        TInitialAll[iDataSet,icondition,:,:]=TInitial

        aux=rand()
        PiInitial[iDataSet,icondition,1]=aux
        PiInitial[iDataSet,icondition,2]=1-aux

        #XInitial[iDataSet,icondition,:]=[1.338313101716786, 1.714232602213121, 3.867537840921497, 3.9042024199430245, 4.752198682714197, 2.332897738999492, 3.4311984203851242, -3.868925326136652]
        #XInitial[iDataSet,icondition,:]=x
        #println("XInitial:", XInitial[iDataSet,icondition,:])
        #println("Tinitial: ", TInitial)
        #println("PiInitial: ", PiInitial[iDataSet,icondition,:])


        PNew,TNew,PiNew,Ll[iDataSet,icondition],ParamFit[iDataSet,icondition,:],xfit=fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,XInitial[iDataSet,icondition,:],lower,upper,TInitial,PiInitial[iDataSet,icondition,:],PossibleOutputs,consts,y)

        if xfit!=xfit || TNew!=TNew
            ConfideceIntervals[iDataSet,icondition,:]=-7*ones(length(ConfideceIntervals[iDataSet,icondition,:]))
        else
            ConfideceIntervals[iDataSet,icondition,:]=ComputeConfidenceIntervals(stim,delays,idelays,choices,past_choices,past_rewards,args,xfit,lower,upper,TNew,PiNew,PossibleOutputs,consts,y)
        end
   end

end

#println("pollas")


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



#filename_save="minimize_sigma_muk_c2_pdwdw_pbiasbias_NDataSet"*string(NDataSets)*"Ntrials"*string(Ntrials)*".jld"

filename_save="minimize_FullModel_NDataSet"*string(NDataSets)*"Ntrials"*string(Ntrials)*".jld"

save(path_results*filename_save,"x",x,"args",args,"y",y,"consts",consts,"XInitial",XInitial,"Ll",Ll,
"PiInitial",PiInitial,"TInitialAll",TInitialAll,"ConfideceIntervals",ConfideceIntervals,
"LlOriginal",LlOriginal,"T",T,"ParamFit",ParamFit)
