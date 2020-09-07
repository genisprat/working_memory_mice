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


P=[0.8 0.2
 0.6 0.4]

T=[0.9 0.1
0.1 0.9]


###### create data ####
Ntrials=1000
state=zeros(Int,Ntrials+1)
choice=zeros(Int,Ntrials)

state[1]=1
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


index1=findall(x->x==1,state[1:end-1])
index2=findall(x->x==2,state[1:end-1])
p1=mean((choice[index1].+1)./2)
p2=mean((choice[index2].+1)./2)



function ProbabilityState(P,T,choice)
    Ntrials=length(choice)
    Nstates=length(P[1,:])
    InitialProb=[0.5,0.5]
    FinalProb=[1,1]
    PFwdState=zeros(Ntrials,Nstates)
    PBackState=zeros(Ntrials,Nstates)
    Norm_coeficcient=zeros(Ntrials)
    #

    for itrial in 1:Ntrials
        for istate in 1:Nstates
            if itrial==1
                aux_sum=InitialProb[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nstates])
            end
            PFwdState[itrial,istate]=aux_sum*P[istate,choice[itrial]]
        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end



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



    Pstate=zeros(Ntrials,Nstates)
    for itrial in 1:Ntrials
        Pstate[itrial,:]=(PFwdState[itrial,:].*PBackState[itrial,:])./(sum(PFwdState[itrial,:].*PBackState[itrial,:]))
    end
    return Norm_coeficcient,PFwdState,PBackState,Pstate
end


Norm_coeficcient,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choice)
figure()


imax=100
plot(choice[1:imax],".k")

plot(state[1:imax],"-k")

plot(PFwdState[1:imax,2].+1,"r.-")
plot(PBackState[1:imax,2].+1,"b.-")
plot(Pstate[1:imax,2].+1,"y.-")

#
# function NegativeLikelihood(P,T,choice)
#
#     Norm_coeficcient,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choice)
#     LogPChoice=zeros(length(choice))
#     #for itrial in 1:length(choice)-1
#         #LogPChoice[itrial]=log( sum(PFwdState[itrial,:].*P[:,choice[itrial+1]]) )
#         #pLogPChoice[itrial]=log(PFwdState[itrial,choice[itrial+1]])
#
#     #end
#     return -sum(log.(Norm_coeficcient))
# end
#
#
#
# function NegativeLikelihood2(P,T,choice)
#
#     Norm_coeficcient,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choice)
#     LogPChoice=zeros(length(choice))
#     for itrial in 1:length(choice)-1
#         LogPChoice[itrial]=log( sum(Pstate[itrial,:].*P[:,choice[itrial]]) )
#         #pLogPChoice[itrial]=log(PFwdState[itrial,choice[itrial+1]])
#
#     end
#     return -sum(LogPChoice)
# end
#





# P1Vector=0.05:0.05:0.95
# P2Vector=0.05:0.05:0.95
# LLpr=zeros(length(P1Vector),length(P2Vector))
# LLpr3=zeros(length(P1Vector),length(P2Vector))
#
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
#         LLpr3[ip1,ip2]=NegativeLikelihood2(Q,T,choice)
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
# figure()
# imshow(LLpr3,origin="lower",extent=[P2Vector[1],P2Vector[end],P1Vector[1],P1Vector[end]],aspect="auto",cmap="hot")
# xlabel("pr2")
# ylabel("pr1")
# plot([P[2,1]],[P[1,1]],"bo")
# a=findall(x->x==minimum(LLpr3),LLpr3)
# plot([ P2Vector[a[1][2]]],[P1Vector[a[1][1]]],"bs")
#
# colorbar()
# show()
#
#
#
hmm = HMM(T, [Categorical(P[1,:]), Categorical(P[2,:])])
alpha, tot = forward(hmm, choice)
beta,tot=backward(hmm,choice)
gamma=posteriors(alpha,beta)

# lll=-likelihoods(hmm,choice,logl=true)
# ll=-loglikelihood(hmm, choice)
# ll3=NegativeLikelihood(P,T,choice)


figure()
plot(alpha[1:imax,1],"r-")
plot(PFwdState[1:imax,1],"k--")
title("alpha")


figure()
plot(beta[1:imax,1],"r-")
plot(PBackState[1:imax,1],"k--")
title("beta")


figure()
plot(gamma[1:imax,1],"r-")
plot(Pstate[1:imax,1],"k--")
title("gamma")


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
