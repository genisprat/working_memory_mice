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



function CreateDataHmmCategorical(P,T,Ntrials,InitialState)
    state=zeros(Int,Ntrials+1)
    choices=zeros(Int,Ntrials)

    state[1]=InitialState
    for itrial in 1:Ntrials
        if rand()<P[state[itrial],1]
            choices[itrial]=1
        else
            choices[itrial]=2
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
    return choices,state
end


function ComputeNegativeLogLikelihood(P,T,choices,InitalP)
    ll,ForwardPass=ForwardPass(P,T,choices,InitalP)
    return ll
end


function ForwardPass(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass and the negative likelihood
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    """
    Nout=length(P[1,1,:])
    Nstate=length(T[1,:])
    PFwdState=zeros(typeof(P[1]),Ntrials,Nout)
    Norm_coeficcient=zeros(typeof(P[1]),Ntrials)

    for itrial in 1:Ntrials
        for istate in 1:Nstate
            if itrial==1
                aux_sum=InitalP[istate]
            else
                aux_sum=sum([ PFwdState[itrial-1,k]*T[k,istate] for k in 1:Nout])
            end
            #println("P: ",P," istate ",istate," choices: ",choices[itrial])
            PFwdState[itrial,istate]=aux_sum*P[itrial,istate,choices[itrial]]

        end
        Norm_coeficcient[itrial]=sum(PFwdState[itrial,:])
        PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))

    end


    return -sum(log.(Norm_coeficcient)),PFwdState
end



function ProbabilityState(P,T,choices,InitalP)
    """
    Function that computes the ForwardPass, the negative log-likelihood,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    """



    FinalProb=[1,1]
    Ntrials=length(choices)
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
    #         PFwdState[itrial,istate]=aux_sum*P[istate,choices[itrial]]
    #     end
    #     PFwdState[itrial,:]=PFwdState[itrial,:]./(sum(PFwdState[itrial,:]))
    # end

    ll,PFwdState=ForwardPass(P,T,choices,InitalP)

    #compute backward pass
    for itrial in 1:Ntrials
        i=Ntrials+1-itrial
        for istate in 1:Nstates
            if itrial==1
                PBackState[i,istate]=FinalProb[istate]
            else
                PBackState[i,istate]=sum(  [ T[istate,k]*P[i+1,k,choices[i+1]]*PBackState[i+1,k] for k in 1:Nstates]   )
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
                norm=norm+PFwdState[itrial,i]*T[i,j]*P[itrial+1,j,choices[itrial+1]]*PBackState[itrial+1,j]
            end
        end

        for i in 1:Nstates
            for j in 1:Nstates
                xi[itrial,i,j]=(PFwdState[itrial,i]*T[i,j]*P[itrial+1,j,choices[itrial+1]]*PBackState[itrial+1,j])/norm
            end
        end
    end




    return Pstate[1,:],PFwdState,PBackState,Pstate,xi,ll
end

function fitBaumWelchAlgorithm(PInitial,TInitial,PiInitial,PossibleOutputs,choices,tol)
    """
    Function that computes the ForwardPass, the negative log-likelihood,
    the backwardPass, the posteriors and xi
    T is the transition Matrix it has dimension Nstates x Nstates

    P is the emission probabilities. Note that in our model the emision
    probabilities change in each trial, they depend on the history and the stimulus.
    Dimension: Ntrials x Nstates x NPossibleOutputs.

    choices the list of outputs it is an integer 1,2.. NpossibleOutputs. Dimension: 1 x Ntrials

    InitialP is the initial probabilities of the states. Dimension 1 x Nstates

    """




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


        auxPi,auxAlpha,auxBeta,auxGamma,auxXi,llAux=ProbabilityState(PIter,TIter,choices,PiIter)


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

        # for i in 1:Nstates
        #     den=sum(auxGamma[:,i])
        #     for j in 1:Nout
        #         index=findall(x->x==PossibleOutputs[j],choices)
        #         PAux[i,j]=sum(auxGamma[index,i])/den
        #     end
        # end






        ##### compute difference between previous and current parameters ####
        #Now with negativelikelihood
        # for i in 1:Nstates*Nstates
        #     delta[i]=TAux[i]-TIter[i]
        # end
        #
        # for i in 1:Nstates*Nout
        #     delta[Nstates*Nstates+i]=PAux[i]-PIter[i]
        # end

        #delta=abs.(delta)
        #println("delta: ", delta)

        #iter=iter+1


        DeltaAux=abs(ll-llAux)

        #update Arrays


        TIter=TAux
        PiIter=PiAux
        PIter=PAux

        ll=llAux

    end

    ll=ComputeNegativeLogLikelihood(PIter,TIter,choices,PiIter)
    #println("iter: ",iter, " ll: ",ll)

    return PIter,TIter,PiIter,ll


end

#
# InitialP=[0.5,0.5]
# Pi,PFwdState,PBackState,Pstate=ProbabilityState(P,T,choices,InitialP)
# figure()
#
#
# imax=100
# plot(choices[1:imax],".k")
#
# plot(state[1:imax],"-k")
#
# plot(PFwdState[1:imax,2].+1,"r.-")
# plot(PBackState[1:imax,2].+1,"b.-")
# plot(Pstate[1:imax,2].+1,"y.-")
#
#
Ntrials=10000
InitialState=1
choices,state=CreateDataHmmCategorical(P,T,Ntrials,InitialState)
index1=findall(x->x==1,state[1:end-2])
index2=findall(x->x==2,state[1:end-2])
p1=mean((choices[index1].-1))
p2=mean((choices[index2].-1))
T12=mean(state[index1.+1].-1)
T22=mean(state[index2.+1].-1)



hmm = HMM(T[:,:], [Categorical(P[1,:]), Categorical(P[2,:])])
alpha,tot=forward(hmm, choices)
beta,ll=backward(hmm,choices)
gamma= posteriors(hmm, choices)

P2=zeros(Ntrials,Nstates,Nout)
for itrial in 1:Ntrials
    P2[itrial,:,:]=P
end
InitalP=[0.5,0.5]
Pi,PFwdState,PBackState,Pstate,xi,ll=ProbabilityState(P2,T,choices,InitalP)


fig,ax=subplots(1,3)
imax=100
ax[1].plot(alpha[1:imax,1],"k-")
ax[1].plot(PFwdState[1:imax,1],"r--")
ylabel("alpha")
ax[2].plot(beta[1:imax,1],"k-")
ax[2].plot(PBackState[1:imax,1],"r--")
ylabel("beta")

ax[3].plot(gamma[1:imax,1],"k-")
ax[3].plot(Pstate[1:imax,1],"r--")
ylabel("gamma")
