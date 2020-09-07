using ForwardDiff
using StatsBase


path="/home/genis/wm_mice/"
include(path*"functions_wm_mice.jl")




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


function ComputeNegativeLogLikelihood(P,T,choices,InitalP)
    ll,alpha=ForwardPass(P,T,choices,InitalP)
    return ll
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



function MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,T,InitialP,args,x,xx,lower,upper)
        # println("hola",x[1])
        # z=zeros(typeof(x[1]),length(x))
        # for i in 1:length(x)
        #     z[i]=x[i]
        # end
        function MaxEmission(y)

            #println("hola2")
            z=zeros(typeof(y[1]),length(x))
            z[:]=x[:]
            #println("hola3")
            z[6]=y[1]
            #println("hola4")
            #println("sigma: ",y[1])
            P=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,z)
            #println("P1: ",P[1]," z: ",z)

            ll=ComputeNegativeLogLikelihood(P,T,choices,InitialP)
            return ll
        end

        #Optim.optimize( MaxEmission,x)

        #res=optimize(MaxEmission, x, LBFGS(); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        #lower=[0.05]
        #upper=[3]
        println("xx: ",xx[1]," lower: ",lower," upper: ",upper)

        res=optimize(MaxEmission,lower,upper, xx, Fminbox(LBFGS(linesearch = BackTracking(order=2))),Optim.Options(show_trace=false); autodiff = :forward)
        #res=optimize(MaxEmission,lower,upper, xx, Fminbox(LBFGS()),Optim.Options(show_trace=true); autodiff = :forward)
        #res=optimize(MaxEmission, xx, LBFGS(); autodiff = :forward)
        x2=zeros(length(x))
        x2[:]=x[:]
        x2[6]=res.minimizer[1]

        Q=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x2)
        #res=optimize(MaxEmission, xx, LBFGS())

        return res.minimizer,Q,res.minimum
end


function fitBaumWelchAlgorithm(stim,delays,idelays,choices,past_choices,past_rewards,args,x,TInitial,PiInitial,PossibleOutputs,tol,xx,lower,upper)

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
    Nout=length(PossibleOutputs)

    delta=ones(Nstates*Nstates+Nout*Nstates)
    iter=1

    TNew=zeros(Nstates,Nstates)
    PNew=zeros(Nstates,Nout)
    PiNew=zero(Nstates)

    POld=ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
    TOld=TInitial[:,:]
    PiOld=PiInitial[:]
    ll=0.0
    DeltaAux=1
    #for iter in 1:Niter-1
    #while all(tol.<delta) aixo esta malament
    while DeltaAux>tol

        #println(PAll[iter,:,:])


        PiNew,AlphaNew,BetaNew,GammaNew,XiNew,llNew=ProbabilityState(POld,TOld,choices,PiOld)


        #############compute new transition matrix##########
        for i in 1:Nstates
            den=sum(GammaNew[:,i])
            for j in 1:Nstates
                TNew[i,j]=sum(XiNew[:,i,j])/den
            end
        end
        ################ New initial conditions ##############

        #Piiter[iter+1,:]=auxGamma[1,:]
        #PiNew=auxGamma[1,:]

        ######## Compute new emission probabilities ##########

        minimizer,PNew,llNew=MaximizeEmissionProbabilities(stim,delays,idelays,choices,past_choices,past_rewards,TOld,POld,args,x,xx,lower,upper)
        xx=minimizer
        println("sigma:", xx[1])
        println("llNew",llNew)
        println("T:", TNew)





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


        DeltaAux=abs(llNew-ll)

        #update Arrays


        TOld=TNew
        PiOld=PiNew
        POld=PNew
        ll=llNew

    end

    ll=ComputeNegativeLogLikelihood(PNew,TNew,choices,PiNew)
    #println("iter: ",iter, " ll: ",ll)
    param=[TNew[1,1],TNew[2,2],xx]

    return PNew,TNew,PiNew,ll,param


end
