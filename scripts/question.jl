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


P=[0.7 0.3
 0.1 0.9]

T=[0.7 0.3
0.1 0.9]

NDataSets=100
Nconditions=20
Nstates=2
Nout=2
Ntrials=1000
LlOriginal=zeros(NDataSets)
LlBest=zeros(NDataSets)

TBest=zeros(NDataSets,Nstates,Nstates)
PiBest=zeros(NDataSets,Nstates)
PBest=zeros(NDataSets,Nstates,Nout)
TFinal=zeros(Nconditions,Nstates,Nstates)
PiFinal=zeros(Nconditions,Nstates)
PFinal=zeros(Nconditions,Nstates,Nout)
Ll=zeros(Nconditions)


for idata in 1:NDataSets
    println("iDataSet: ",idata)


    #create a new hmm object
    hmm = HMM(T[:,:], [Categorical(P[1,:]), Categorical(P[2,:])])
    println(hmm.A)
    println(hmm.B[1].p)
    println(hmm.B[2].p)
    choice,state=rand(hmm, Ntrials, seq = true) #create synthetic data
    LlOriginal[idata]=-loglikelihood(hmm,choice) #LL of the original parameters


    for icondition in 1:Nconditions
        #randomize initial condition
        aux=rand(4)
        hmm.A[1,1]=aux[1]
        hmm.A[1,2]=1-aux[1]
        hmm.A[2,1]=aux[2]
        hmm.A[2,2]=1-aux[2]

        hmm.B[1].p[1]=aux[3]
        hmm.B[1].p[2]=1-aux[3]
        hmm.B[2].p[1]=aux[4]
        hmm.B[2].p[2]=1-aux[4]



        hmm2,history=fit_mle(hmm, choice) #fit
        #println(hmm2.A)
        TFinal[icondition,:,:]=hmm2.A
        PFinal[icondition,1,:]=hmm2.B[1].p
        PFinal[icondition,2,:]=hmm2.B[2].p
        PiFinal[icondition,:]=hmm2.a
        Ll[icondition]=-loglikelihood(hmm2,choice)

    end
    #best parameters fit of the Nconditions initial conditions
    imin=findall(x->x==minimum(Ll),Ll)[1]

    TBest[idata,:,:]=TFinal[imin,:,:]
    PBest[idata,:,:]=PFinal[imin,:,:]
    PiBest[idata,:,:]=PiFinal[imin,:,:]
    LlBest[idata]=Ll[imin]



end


figure()
plot(TBest[:,1,1],1.0 .+rand(NDataSets)./3,"k.")
plot(TBest[:,2,2],2.0 .+rand(NDataSets)./3,"k.")
plot(PBest[:,1,1],3.0 .+rand(NDataSets)./3,"k.")
plot(PBest[:,2,2],4.0 .+rand(NDataSets)./3,"k.")

xlabel("Fitted Param Value")
ylabel("Param")

plot([T[1,1],T[1,1]],[1,3],"r-")
plot([T[2,2],T[2,2]],[1,3],"r-")

plot([P[1,1],P[1,1]],[3,5],"b--")
plot([P[2,2],P[2,2]],[3,5],"b--")
# ax[itrial].plot([P[1,2],P[1,2]],[3,5],"b--")
# ax[itrial].plot([P[2,1],P[2,1]],[3,5],"b--")

# ax2[itrial].plot(LlOriginal,"r.")
# ax2[itrial].plot(LlBest,"k.")






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
