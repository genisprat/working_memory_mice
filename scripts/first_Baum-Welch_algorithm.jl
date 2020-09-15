using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches

const epsilon=1e-8

using Distributions
using HMMBase
###### definitions ####


P=[0.9 0.1
 0.1 0.9]

T=[0.8 0.2
0.1 0.9]


###### create data ####
Ntrials=1000
InitialState=1
function CreateDataHmmCategorical(P,T,Ntrials,InitialState)
    state=zeros(Int,Ntrials+1)
    choice=zeros(Int,Ntrials)

    state[1]=InitialState
    for itrial in 1:Ntrials
        if rand()<P[state[itrial],1]
            choice[itrial]=1
        else
            choice[itrial]=2
        end

        if rand()<T[state[itrial],state[itrial]]
            state[itrial+1]=state[itrial]
        else
            if state[itrial]==1
                state[itrial+1]=2
            else
                state[itrial+1]=1

            end
        end
    end
    return choice,state
end


function ComputeNegativeLogLikelihood(P,T,choice,InitalP)

    PFwdState=zeros(typeof(P[1]),Ntrials,length(P[1,:]))
    Norm_coeficcient=zeros(typeof(P[1]),Ntrials)

    for itrial in 1:Ntrials
        for istate in 1:length(P[1,:])
            if itrial==1
                aux_sum=InitalP[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:length(P[1,:])])
            end
            PFwdState[itrial,istate]=aux_sum*P[istate,choice[itrial]]
        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end


    return -sum(log.(Norm_coeficcient))
end


function ForwardPass(P,T,choice,InitalP)

    PFwdState=zeros(typeof(P[1]),Ntrials,length(P[1,:]))
    Norm_coeficcient=zeros(typeof(P[1]),Ntrials)

    for itrial in 1:Ntrials
        for istate in 1:length(P[1,:])
            if itrial==1
                aux_sum=InitalP[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:length(P[1,:])])
            end
            #println("P: ",P," istate ",istate," choice: ",choice[itrial])
            PFwdState[itrial,istate]=aux_sum*P[istate,choice[itrial]]

        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end


    return -sum(log.(Norm_coeficcient)),PFwdState
end



function ProbabilityState(P,T,choice,InitalP)

    FinalProb=[1,1]
    Ntrials=length(choice)
    Nstates=length(T[1,:])
    PFwdState=zeros(Ntrials,Nstates)
    PBackState=zeros(Ntrials,Nstates)
    ll=0
    # for itrial in 1:Ntrials
    #     #compute forward pass
    #     for istate in 1:Nstates
    #         if itrial==1
    #             aux_sum=InitalP[istate]
    #         else
    #             aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nstates])
    #         end
    #         PFwdState[itrial,istate]=aux_sum*P[istate,choice[itrial]]
    #     end
    #     PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))
    # end

    ll,PFwdState=ForwardPass(P,T,choice,InitalP)

    #compute backward pass
    for itrial in 1:Ntrials
        i=Ntrials+1-itrial
        for istate in 1:Nstates
            if itrial==1
                PBackState[i,istate]=FinalProb[istate]
            else
                PBackState[i,istate]=sum(  [ T[istate,k]*P[k,choice[i+1]]*PBackState[i+1,k] for k in 1:Nstates]   )
            end
        end
        PBackState[i,:]=PBackState[i,:]./sum(PBackState[i,:])
    end


    #compute merge state probabilities
    Pstate=zeros(typeof(P[1]),Ntrials,Nstates)
    for itrial in 1:Ntrials
        Pstate[itrial,:]=(PFwdState[itrial,:].*PBackState[itrial,:])./(sum(PFwdState[itrial,:].*PBackState[itrial,:]))
    end



    #compute transition prob at time t from i to j
    xi=zeros( typeof(P[1]), Ntrials-1, Nstates,Nstates )
    for itrial in 1:Ntrials-1
        #####cumpute normalitzation#####
        norm=0
        for i in 1:Nstates
            for j in 1:Nstates
                norm=norm+PFwdState[itrial,i]*T[i,j]*P[j,choice[itrial+1]]*PBackState[itrial+1,j]
            end
        end

        for i in 1:Nstates
            for j in 1:Nstates
                xi[itrial,i,j]=(PFwdState[itrial,i]*T[i,j]*P[j,choice[itrial+1]]*PBackState[itrial+1,j])/norm
            end
        end
    end




    return Pstate[1,:],PFwdState,PBackState,Pstate,xi,ll
end

function fitBaumWelchAlgorithm(PInitial,TInitial,PiInitial,PossibleOutputs,choice,tol)
    # local TInitial
    # local PInitial
    # local PiInitial
    Nstates=length(TInitial[1,:])
    Nout=length(PInitial[1,:])

    delta=ones(Nstates*Nstates+Nout*Nstates)
    iter=1

    TAux=zeros(Nstates,Nstates)
    PAux=zeros(Nstates,Nout)
    PiAux=zero(Nstates)

    PIter=PInitial[:,:]
    TIter=TInitial[:,:]
    PiIter=PiInitial[:]
    ll=0.0
    DeltaAux=1
    #for iter in 1:Niter-1
    #while all(tol.<delta) aixo esta malament
    while DeltaAux>tol

        #println(PAll[iter,:,:])


        auxPi,auxAlpha,auxBeta,auxGamma,auxXi,llAux=ProbabilityState(PIter,TIter,choice,PiIter)


        #############compute new transition matrix##########
        for i in 1:Nstates
            den=sum(auxGamma[:,i])
            for j in 1:Nstates
                TAux[i,j]=sum(auxXi[:,i,j])/den
            end
        end
        ################ New initial conditions ##############

        #Piiter[iter+1,:]=auxGamma[1,:]
        PiAux=auxGamma[1,:]

        ######## Compute new emission probabilities ##########

        for i in 1:Nstates
            den=sum(auxGamma[:,i])
            for j in 1:Nout
                index=findall(x->x==PossibleOutputs[j],choice)
                PAux[i,j]=sum(auxGamma[index,i])/den
            end
        end
        ##### compute difference between previous and current parameters ####
        for i in 1:Nstates*Nstates
            delta[i]=TAux[i]-TIter[i]
        end

        for i in 1:Nstates*Nout
            delta[Nstates*Nstates+i]=PAux[i]-PIter[i]
        end

        #delta=abs.(delta)
        #println("delta: ", delta)

        iter=iter+1

        #update Arrays
        TIter=TAux
        PiIter=PiAux
        PIter=PAux

        DeltaAux=abs(ll-llAux)
        ll=llAux

    end

    ll=ComputeNegativeLogLikelihood(PIter,TIter,choice,PiIter)
    #println("iter: ",iter, " ll: ",ll)

    return PIter,TIter,PiIter,ll


end

#
# InitialP=[0.5,0.5]
# Pi,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choice,InitialP)
# figure()
#
#
# imax=100
# plot(choice[1:imax],".k")
#
# plot(state[1:imax],"-k")
#
# plot(PFwdState[1:imax,2].+1,"r.-")
# plot(PBackState[1:imax,2].+1,"b.-")
# plot(Pstate[1:imax,2].+1,"y.-")
#
#
Ntrials=10000
choice,state=CreateDataHmmCategorical(P,T,Ntrials,InitialState)
index1=findall(x->x==1,state[1:end-2])
index2=findall(x->x==2,state[1:end-2])
p1=mean((choice[index1].-1))
p2=mean((choice[index2].-1))
T12=mean(state[index1.+1].-1)
T22=mean(state[index2.+1].-1)



Niter=1000

Nstates=length(T[1,:])
Nout=length(P[1,:])
PossibleOutputs=[1,2]


NDataSets=100
Ntrials=1000
LlOriginal=zeros(NDataSets)
LlBest=zeros(NDataSets)

TBest=zeros(NDataSets,Nstates,Nstates)
PiBest=zeros(NDataSets,Nstates)
PBest=zeros(NDataSets,Nstates,Nout)

for idata in 1:NDataSets
    println("iDataSet: ",idata)
    choice,state=CreateDataHmmCategorical(P,T,Ntrials,InitialState)
    Nconditions=200

    TFinal=zeros(Nconditions,Nstates,Nstates)
    PiFinal=zeros(Nconditions,Nstates)
    PFinal=zeros(Nconditions,Nstates,Nout)
    Ll=zeros(Nconditions)
    aux=rand(5)
    PInitial=[aux[1] 1-aux[1] ; aux[2] 1-aux[2]]
    TInitial=[aux[3] 1-aux[3] ; aux[4] 1-aux[4]]
    PiInitial=[aux[5] 1-aux[5]]

    LlOriginal[idata]=ComputeNegativeLogLikelihood(P,T,choice,[1,0])



    for icondition in 1:Nconditions

        #println(icondition)
        tol=1e-3
        PFit,TFit,PiFit,ll=fitBaumWelchAlgorithm(PInitial,TInitial,PiInitial,PossibleOutputs,choice,tol)

        TFinal[icondition,:,:]=TFit
        PiFinal[icondition,:]=PiFit
        PFinal[icondition,:,:]=PFit
        Ll[icondition]=ll

    end

    imin=findall(x->x==minimum(Ll),Ll)[1]

    TBest[idata,:,:]=TFinal[imin,:,:]
    PBest[idata,:,:]=PFinal[imin,:,:]
    PiBest[idata,:,:]=PiFinal[imin,:,:]
    LlBest[idata]=Ll[imin]

end


filename_save="/home/genis/wm_mice/synthetic_data/FitHmmCategorical_Ntrials"*string(Ntrials)*".jld"
JLD.save(filename_save,"TBest",TBest,"PBest",PBest,"PiBest",PiBest,"LlOriginal",LlOriginal,"LlBest",LlBest,"P",P,"T",T)



####################### SAME USING HmmBase ########################
# NDataSets=100
# Nconditions=20
#
# Ntrials=1000
# LlOriginal=zeros(NDataSets)
# LlBest=zeros(NDataSets)
#
# TBest=zeros(NDataSets,Nstates,Nstates)
# PiBest=zeros(NDataSets,Nstates)
# PBest=zeros(NDataSets,Nstates,Nout)
# TFinal=zeros(Nconditions,Nstates,Nstates)
# PiFinal=zeros(Nconditions,Nstates)
# PFinal=zeros(Nconditions,Nstates,Nout)
# Ll=zeros(Nconditions)
#
#
#
#
# hmm = HMM(T[:,:], [Categorical(P[1,:]), Categorical(P[2,:])])
#
# for idata in 1:NDataSets
#     println("iDataSet: ",idata)
#
#
#     #choice,state=CreateDataHmmCategorical(P,T,Ntrials,InitialState)
#     hmm = HMM(T[:,:], [Categorical(P[1,:]), Categorical(P[2,:])])
#     println(hmm.A)
#     println(hmm.B[1].p)
#     println(hmm.B[2].p)
#     choice,state=rand(hmm, Ntrials, seq = true)
#     LlOriginal[idata]=ComputeNegativeLogLikelihood(P,T,choice,[1,0])
#
#
#     for icondition in 1:Nconditions
#         aux=rand(4)
#         hmm.A[1,1]=aux[1]
#         hmm.A[1,2]=1-aux[1]
#         hmm.A[2,1]=aux[2]
#         hmm.A[2,2]=1-aux[2]
#
#         hmm.B[1].p[1]=aux[3]
#         hmm.B[1].p[2]=1-aux[3]
#         hmm.B[2].p[1]=aux[4]
#         hmm.B[2].p[2]=1-aux[4]
#
#
#
#         hmm2,history=fit_mle(hmm, choice)
#         #println(hmm2.A)
#         TFinal[icondition,:,:]=hmm2.A
#         PFinal[icondition,1,:]=hmm2.B[1].p
#         PFinal[icondition,2,:]=hmm2.B[2].p
#         PiFinal[icondition,:]=hmm2.a
#         Ll[icondition]=-loglikelihood(hmm2,choice)
#
#     end
#
#     imin=findall(x->x==minimum(Ll),Ll)[1]
#
#     TBest[idata,:,:]=TFinal[imin,:,:]
#     PBest[idata,:,:]=PFinal[imin,:,:]
#     PiBest[idata,:,:]=PiFinal[imin,:,:]
#     LlBest[idata]=Ll[imin]
#
#
#
# end
#
# filename_save="/home/genis/wm_mice/synthetic_data/HmmBaseFitHmmCategorical_Ntrials"*string(Ntrials)*".jld"
# JLD.save(filename_save,"TBest",TBest,"PBest",PBest,"PiBest",PiBest,"LlOriginal",LlOriginal,"LlBest",LlBest,"P",P,"T",T)
#
#



# Algorithm with traces

# while all(tol.<delta)
#     #println(PAll[iter,:,:])
#     global iter,delta,tol
#     auxPi,auxAlpha,auxBeta,auxGamma,auxXi=ProbabilityState(PAll[iter,:,:],TAll[iter,:,:],choice,PiAll[iter,:])
#
#
#     #############compute new transition matrix##########
#     for i in 1:Nstates
#         den=sum(auxGamma[:,i])
#         for j in 1:Nstates
#             TAll[iter+1,i,j]=sum(auxXi[:,i,j])/den
#         end
#     end
#     ################ New initial conditions ##############
#
#     #Piiter[iter+1,:]=auxGamma[1,:]
#     PiAll[iter+1,:]=auxGamma[1,:]
#
#     ######## Compute new emission probabilities ##########
#
#     for i in 1:Nstates
#         den=sum(auxGamma[:,i])
#         for j in 1:Nout
#             index=findall(x->x==PossibleOutputs[j],choice)
#             PAll[iter+1,i,j]=sum(auxGamma[index,i])/den
#         end
#     end
#
#     for i in 1:Nstates*Nstates
#         delta[i]=TAll[iter+1,:,:][i]-TAll[iter,:,:][i]
#     end
#
#     for i in 1:Nstates*Nout
#         delta[Nstates*Nstates+i]=PAll[iter+1,:,:][i]-PAll[iter,:,:][i]
#     end
#
#     delta=abs.(delta)
#     iter=iter+1
#
#
# end

#
# figure()
#
# plot(TAll[1:iter,1,1])
# plot(TAll[1:iter,2,2])
#
#
# figure()
#
# plot(PAll[1:iter,1,1])
# plot(PAll[1:iter,2,2])
#

# function NegativeLikelihood(P,T,choice,InitalP)
#
#     Norm_coeficcient,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choice,InitalP)
#     LogPChoice=zeros(length(choice))
#     #for itrial in 1:length(choice)-1
#         #LogPChoice[itrial]=log( sum(PFwdState[itrial,:].*P[:,choice[itrial+1]]) )
#         #pLogPChoice[itrial]=log(PFwdState[itrial,choice[itrial+1]])
#
#     #end
#     return -sum(log.(Norm_coeficcient))
# end





# P1Vector=0.05:0.05:0.95
# P2Vector=0.05:0.05:0.95
# LLpr=zeros(length(P1Vector),length(P2Vector))
# Q=zeros(2,2)
# for ip1 in 1:length(P1Vector)
#     pr1=P2Vector[ip1]
#     Q[1,1]=pr1
#     Q[1,2]=1-pr1
#     for ip2 in 1:length(P2Vector)
#         pr2=P2Vector[ip2]
#         Q[2,1]=pr2
#         Q[2,2]=1-pr2
#         #println("Q: ",Q)
#         LLpr[ip1,ip2]=NegativeLikelihood(Q,T,choice)
#     end
# end
#
#
# figure()
# imshow(LLpr,origin="lower",extent=[P2Vector[1],P2Vector[end],P1Vector[1],P1Vector[end]],aspect="auto",cmap="hot")
# xlabel("pr2")
# ylabel("pr1")
# plot([P[2,1]],[P[1,1]],"bo")
# a=findall(x->x==minimum(LLpr),LLpr)
# plot([ P2Vector[a[1][2]]],[P1Vector[a[1][1]]],"bs")
#
# colorbar()
# show()
#
#
# hmm = HMM(T, [Categorical(P[1,:]), Categorical(P[2,:])])
# probs, tot = forward(hmm, choice)
# lll=-likelihoods(hmm,choice,logl=true)
# ll=-loglikelihood(hmm, choice)
# ll3=NegativeLikelihood(P,T,choice)
#
#
# a=zeros(length(choice))
# for itrial in 1:length(choice)
#     a[itrial]=lll[itrial,choice[itrial]]
# end
# ll2=sum(a)
#
# figure()
#
# plot(probs[1:imax,1],"r-")
# plot(probs[1:imax,1],"k--")
#
#
#
#
#
#
# P1Vector=0.05:0.05:0.95
# P2Vector=0.05:0.05:0.95
# LLpr2=zeros(length(P1Vector),length(P2Vector))
# Q=zeros(2,2)
# for ip1 in 1:length(P1Vector)
#     pr1=P2Vector[ip1]
#     Q[1,1]=pr1
#     Q[1,2]=1-pr1
#     for ip2 in 1:length(P2Vector)
#         pr2=P2Vector[ip2]
#         Q[2,1]=pr2
#         Q[2,2]=1-pr2
#         #println("Q: ",Q)
#         hmm = HMM(T, [Categorical(Q[1,:]), Categorical(Q[2,:])])
#         LLpr2[ip1,ip2]=-loglikelihood(hmm, choice)
#
#     end
# end
#
#
#
# figure()
# imshow(LLpr2,origin="lower",extent=[P2Vector[1],P2Vector[end],P1Vector[1],P1Vector[end]],aspect="auto",cmap="hot")
# xlabel("pr2")
# ylabel("pr1")
# plot([P[2,1]],[P[1,1]],"bo")
# a=findall(x->x==minimum(LLpr2),LLpr2)
# plot([ P2Vector[a[1][2]]],[P1Vector[a[1][1]]],"bs")
#
# colorbar()
# show()
