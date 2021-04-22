
using SpecialFunctions
#using PyPlot
using ForwardDiff
using StatsBase

const epsilon=1e-12

function make_dict2(args,x)
    nargs = 0
    for i in [1:length(args);]
        if typeof(args[i])==String # if the entry in args is a string, then there's one corresponding scalar entry in x0
            nargs += 1
        else
            nargs += args[i][2]    # otherwise, the entry in args should be a  [varnamestring, nvals] vector,
            # indicating that the next nvals entries in x0 are all a single vector, belonging to variable
            # with name varnamestring.
        end
    end
    if nargs != length(x)

        error("Oy! args and x must indicate the same total number of variables!")

    end

    i=1
    param=Dict()
    for  (iarg,arg) in enumerate(args)
        if typeof(arg)==String
            a=x[i:i]
            param[arg]=x[i:i][1]
            i+=1
        else
            j=arg[2]
            param[arg[1]]=x[i:i+j-1]
            i+=j
        end
    end

    return param
end


function make_dict(parameters,y,constans=0,c=0)
    if constans==0
        args=parameters
        x=y
    else
        args=vcat(parameters,constans)
        x=vcat(y,c)
    end
    nargs = 0
    for i in [1:length(args);]
        if typeof(args[i])==String # if the entry in args is a string, then there's one corresponding scalar entry in x0
            nargs += 1
        else
            nargs += args[i][2]    # otherwise, the entry in args should be a  [varnamestring, nvals] vector,
            # indicating that the next nvals entries in x0 are all a single vector, belonging to variable
            # with name varnamestring.
        end
    end
    if nargs != length(x)

        error("Oy! args and x must indicate the same total number of variables!")
    end

    i=1
    param=Dict()
    for  (iarg,arg) in enumerate(args)
        if typeof(arg)==String
            a=x[i:i]
            param[arg]=x[i:i][1]
            i+=1
        else
            j=arg[2]
            param[arg[1]]=x[i:i+j-1]
            i+=j
        end
    end

    return param
end









function initial_transition(mu,c2,xL,xR,x0,sigma)

    if c2>0
        c2_sqrt=sqrt(c2)
    else
        c2_sqrt=NaN
        error("c2 negative", c2, "mu:",mu,"xL ",xL,"xR: ",xR,"x0: ", x0, "sigma: ",sigma)
    end
    if x0>xR
        return 1.0
    elseif x0<xL
        return 0.0
    else

        return ( erf(c2_sqrt/sigma*(x0+mu/c2)) - erf(c2_sqrt/sigma*(xL+mu/c2)) )/(  erf(c2_sqrt/sigma*(xR+mu/c2)) - erf(c2_sqrt/sigma*(xL+mu/c2)) )
    end

end

function roots_DW(coef)
    """
    Analytical positions of the L and R attractor as well as
    the intermediate stable state.
    d=-mu
    c=-c2
    a=c4
    """

    d=-coef[1]
    c=-coef[2]
    a=coef[3]
    roots=zeros(typeof(d),3)
    delta=-4*a*c^3-27*(a^2)*d^2
    #roots_aux=zeros(typeof(roots[1]),3)
    if sign(delta)==-1
        roots[1]=-sign(d)
        roots[2]=NaN
        roots[3]=NaN
        error("delta: ", delta," ",d," c ",c," a: ", a)
    elseif coef[1]==0
        roots[1]=-sqrt(coef[2]/coef[3])
        roots[2]=0
        roots[3]=sqrt(coef[2]/coef[3])
    else
        p=c/a
        q=d/a
        aux_sqrt=(27.0/4.0)*(q^2)/(-p^3)
        if aux_sqrt>0
            phi=acos( sqrt( aux_sqrt  ) )
        else
            error("phi negative", coef[1]," ",coef[2]," ",coef[3])
        end
        # k=np.array(range(3))
        K=[0,1,2]
        if p<0

            k0=2.0*sqrt(-p/3.0)*cos( phi/3.0+2.0943951*0)
            k1=2.0*sqrt(-p/3.0)*cos( phi/3.0+2.0943951*1)
            k2=2.0*sqrt(-p/3.0)*cos( phi/3.0+2.0943951*2)
        else
            error("p negative", coef[1]," ",coef[2]," ",coef[3])
            k0=NaN
            k1=NaN
            k2=NaN
        end
        #println("typeof k: ", typeof(k0))

        # for k in K
        #     roots_aux[k+1]=2*sqrt(-p/3.0)*cos( phi/3.0+2.0943951*k)
        #     println("roots aux: ",roots_aux)
        # end
        #sign_aux=(q*q)/(q*sqrt(q*q))
        #println("typeof sign", typeof(sign_aux),sign_aux)
        sign_aux=sign(q)
        #println("signa q: ",sign_aux)
        #println("")

        if sign_aux>0
            #return [sign*roots[1],sign*roots[2],sign*roots[0]] #id check the order is always like this we do not need to sort.
            roots[1]=sign_aux*k1#roots_aux[2]
            roots[2]=sign_aux*k2#roots_aux[3]
            roots[3]=sign_aux*k0#roots_aux[1]
        elseif sign_aux==0
            #return [roots[1],roots[2],roots[0]]
            #return [roots[1],roots[2],roots[0]]
            roots[1]=k1#roots_aux[2]
            roots[2]=k2#roots_aux[3]
            roots[3]=k0#roots_aux[1]

        else
            #return [sign*roots[0],sign*roots[2],sign*roots[1]]
            roots[1]=k0*sign_aux#*roots_aux[1]
            roots[2]=k2*sign_aux#*roots_aux[3]
            roots[3]=k1*sign_aux#*roots_aux[2]
        end
    end
    return roots
end

function max_mu(coef)
    c=-coef[2]
    a=coef[3]
    aux=-4.0*c^3/(27.0*a)
    if aux>0
        max_mu2=sqrt(aux)
    else
        max_mu2=NaN
        error("Negative sqrt in max_mu",coef[1]," ",coef[2]," ",coef[3])
    end
    return max_mu2
end



function potential_DW(x,coef)
    """
    DW potential at x with coeficients coef
    coef[1]=mu
    coef[2]=c2
    coef[3]=c4
    """
    return -coef[1]*x-0.5*coef[2]*x^2+0.25*coef[3]*x^4
end



function potential_DW_prima2(x,coef)
    """
    First derivative of the DW potential at x with coeficients coef
    coef[1]=mu
    coef[2]=c2
    coef[3]=c4
    """
    return -coef[2]+3.0*coef[3]*x^2

end


function Transition_rates(coef,xL,xM,xR,sigma)
    """
    Transistions rates from R to L (kxLxR) and from L to R (kxRxL).
    coef is an array with the potential coeficiant mu, c2 and c4
    xL and xR are the positions of the attractors
    xM is the position of maximum between the two attractors
    sigma is the level of noise
    """
    D=1.0*sigma^2/2.0

    delta_xLxM=potential_DW(xM,coef)- potential_DW(xL,coef)
    delta_xRxM=potential_DW(xM,coef)- potential_DW(xR,coef)

    aux_kxRxL=sqrt(abs(potential_DW_prima2(xM,coef)*potential_DW_prima2(xL,coef)))/(2.0*pi)
    aux_kxLxR=sqrt(abs(potential_DW_prima2(xM,coef)*potential_DW_prima2(xR,coef)))/(2.0*pi)
    # print delta_xRxM,delta_xLxM,D
    kxLxR=aux_kxLxR*exp(-delta_xRxM/D)
    kxRxL=aux_kxRxL*exp(-delta_xLxM/D)
    if kxLxR!=kxLxR || kxRxL!=kxRxL
        error("NaN in transitions rates:  coef: ",coef, " sigma: ",sigma," k:", kxLxR," k:", kxRxL, "D: ",D, "kxLxR: ",kxLxR," kxRxL: ",kxRxL," delta_xRxM: ",delta_xRxM," delta_xRxM: ",delta_xRxM)
    end
    # print kxLxR,kxRxL
    return kxRxL,kxLxR
    # return kxRxL,kxLxR
end


function Transition_probabilites(coef,xL,xM,xR,sigma,t)
    """
    Computes the transistion probability to reamin in R
    after t and the probabilituy to find the system in L
    given that it starts in R after a time t.
    Important t is in units of tau t=Tframe/tau where Tframe is
    the time in ms
    """

    kxRxL,kxLxR=Transition_rates(coef,xL,xM,xR,sigma)
    #println("Trans: ",kxRxL," ",kxLxR)
    # print coef,xL,xM,xR,sigma,t

    prs=kxRxL/(kxLxR+kxRxL)

    if prs!=prs
        if coef[1]>0
            prs=1.0 #The limit of prs when sigma tends to 0 depends goes to 1 if mu>0
        else
            prs=0.0
        end
    end
    aux=exp(-(kxLxR+kxRxL)*t)
    if aux!=aux
        maximum_mu=max_mu(coef)
        if coef[1]>maximum_mu
            prr=1.0 -epsilon
            prl=1.0 -epsilon#The limit of prs when sigma tends to 0 depends goes to 1 if mu>0
        elseif coef[1]<-maximum_mu
            prr=0. +epsilon
            prl=0. +epsilon#The limit of prs when sigma tends to 0 depends goes to 1 if mu>0

        else
            prr=1.0 -epsilon
            prl=0.0 +epsilon
        end
    else
        prr=prs+aux*(1-prs)
        prl=prs*(1-aux)

    end
    #println(aux)


    if prr!=prr || prl!=prl || prs!=prs
        error("NaN in transitions probabilities:", " prs:",prs, " aux:",aux, " prr:",prr," prl:",prl," coef: ",coef, " sigma: ",sigma," k:", kxLxR," k:", kxRxL)
    end
    # except OverflowError:
    #     print -(kxLxR+kxRxL)*t,t,kxLxR,'putaaaaaaaaaaaaa'
    #     prr=prs
    #     prl=prs

return prr,prl
end


function PR_1stim(coef,sigma,x0,mu_b,delay)
    mu=coef[1]+mu_b
    c2=coef[2]
    c4=coef[3]

    maximum_mu=max_mu(coef)
    pr=0
    pr0=0
    if mu>=maximum_mu || mu_b>=maximum_mu
        pr0=1.0
    elseif mu <= -maximum_mu || mu_b <= -maximum_mu
        pr0=0.0
        #println("puta pr0=0")
    else
        #println("puta")
        roots=roots_DW([mu,c2,c4])
        #println(roots," c2",c2," c4",c4)
        pr0=initial_transition(mu,c2,roots[1],roots[3],x0,sigma)
        if pr0!=pr0
            error("Pr0 is NaN",roots)
        end
    end
    #println("hol")
    if mu_b >= maximum_mu
        pr=1.0
    elseif  mu_b <= -maximum_mu
        pr=0.0
    else
        coef_wm=[mu_b,c2,c4]
        #println("coef",coef_wm)
        roots=roots_DW(coef_wm)
        #println(roots," c22",c2," c44",c4)
        prr,prl=Transition_probabilites(coef_wm,roots[1],roots[2],roots[3],sigma,delay)
        pr=pr0*prr+(1-pr0)*prl
    end
    #For numerical reasons I keep pr between epsilon and 1-epsilon
    pr=maximum([epsilon,pr])
    pr=minimum((1-epsilon,pr))
    return pr

end



function history_bias_module_1stim(beta_w,beta_l,tau_w,tau_l,past_choices,past_rewards)
    t=0:length(past_choices)-1
    x=sum(beta_w.*exp.(-t./tau_w).*past_rewards.*past_choices)+
      +sum(beta_l*exp.(-t./tau_l).*(1.0.-past_rewards).*past_choices)

    return 1/(1+exp(-x))
end



function repeating_module_1stim(beta_r,tau_r,past_choices)
    t=0:length(past_choices)-1
    x=sum(beta_r.*exp.(-t./tau_r).*past_choices)
    return 1/(1+exp(-x))
end


#
# a=[1,-1,1,-1,1]
# b=[1,0,1,1,0]
# beta_w=0.2
# beta_l=3
# tau_w=10
# tau_l=3
#history_bias_module_1stim(beta_w,beta_l,tau_w,tau_l,a,b)



function update_PDw(PDwDw,PBiasBias,PDw,PrDw,PrBias,c)
    PDw2=PDw*PDwDw + (1-PDw)*(1-PBiasBias) #Prob to be in the Dw module in the current trial
    PBias2=(1-PDw)*PBiasBias + PDw*(1-PDwDw) #Prob to be in the Bias module in the current trial

    if c==1
        PDw3=PDw2*PrDw  ###Prob of each state given the decicion in the current ###
        PBias3=PBias2*PrBias
    else
        PDw3=PDw2*(1-PrDw)
        PBias3=PBias2*(1-PrBias)
    end

    Norm=PDw3+PBias3

    return PDw3/Norm #return the prob of being in the Dw module in the next trial#
end


function ComputePR(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts=0,y=0)
    """
    Computes PR for all trials
    stim: It is an Array with value 1 if stimulus ir R and 2 if stimulus is L. 1 x Ntrials
    delays: Array with all possibles delays in the data
    idelays: Array with the index of the delay corresponding to each trial. 1 x Ntrials
    past_choices is a matrix with the past_choices for each trial.  Ntrials x N_past_choices
    where N_past_choices is the number of past choices that the model is using.
    past_rewards: Same as past_choices but with the rewards
    args: Name of parameters
    x: List of parameters
    """

    Ntrials=length(stim)
    param=make_dict(args,x,consts,y)
    PDw=zeros(typeof(x[1]),Ntrials+1)
    PDw[1]=0.5
    Pr=zeros(typeof(x[1]),Ntrials)
    PrBias=zeros(typeof(x[1]),Ntrials)
    PrDw=zeros(typeof(x[1]),2,length(delays))
    #Compute prob of Right for each combination of stimulus and delay###
    MU=[-1,1]
    for idelay in 1:length(delays)
        coef=[MU[1]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[1,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])


        coef=[MU[2]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[2,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])


    end



    for itrial in 1:Ntrials

        #PrBias[itrial]=history_bias_module_1stim(param["beta_w"],param["beta_l"],param["tau_w"],param["tau_l"],past_choices[itrial,:],past_rewards[itrial,:])
        PrBias[itrial]=repeating_module_1stim(param["beta_r"],param["tau_r"],past_choices[itrial,:])


        Pr[itrial]=PDw[itrial]*PrDw[stim[itrial],idelays[itrial]]+(1-PDw[itrial])*PrBias[itrial]
        PDw[itrial+1]=update_PDw(param["PDwDw"],param["PBiasBias"],PDw[itrial],PrDw[stim[itrial],idelays[itrial]],PrBias[itrial],choices[itrial])
    end

    return Pr

end

function ComputeEmissionProb(stim,delays,idelays,choices,past_choices,past_rewards,args,x,consts=0,y=0)
    """
    Computes PR for all trials
    stim: It is an Array with value 1 if stimulus ir R and 2 if stimulus is L. 1 x Ntrials
    delays: Array with all possibles delays in the data
    idelays: Array with the index of the delay corresponding to each trial. 1 x Ntrials
    past_choices is a matrix with the past_choices for each trial.  Ntrials x N_past_choices
    where N_past_choices is the number of past choices that the model is using.
    past_rewards: Same as past_choices but with the rewards
    args: Name of parameters
    x: List of parameters
    """

    if x!=x
        error("Nans in x", x)
    end

    Ntrials=length(stim)
    Nout=2
    P=zeros(typeof(x[1]),Ntrials,Nout,Nout)

    param=make_dict(args,x,consts,y)
    if param!=param
        error("Nans in param",param)
    end
    param["sigma"]=sqrt(param["sigma"]^2) #make sure it is not negative in case i do not use FminBox
    param["c2"]=sqrt(param["c2"]^2)
    param["c4"]=sqrt(param["c4"]^2)
    param["mu_k"]=sqrt(param["mu_k"]^2)
    #println(param)
    PrDw=zeros(typeof(x[1]),2,length(delays))
    #Compute prob of Right for each combination of stimulus and delay###

    #println("param: ",param)
    ###### Pr for working memory module, module=1 #################
    MU=[-1,1]
    for idelay in 1:length(delays)
        coef=[MU[1]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[1,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])

        coef=[MU[2]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[2,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])

    end


    for itrial in 1:Ntrials
        pr=PrDw[stim_index[itrial],idelays[itrial]]
        P[itrial,1,1]=pr
        P[itrial,1,2]=1-pr
    end


    ###### Pr for history bias module, module=1 #################

    for itrial in 1:Ntrials

        #pr=history_bias_module_1stim(param["beta_w"],param["beta_l"],param["tau_w"],param["tau_l"],past_choices[itrial,:],past_rewards[itrial,:])
        pr=repeating_module_1stim(param["beta_r"],param["tau_r"],past_choices[itrial,:])

        P[itrial,2,1]=pr
        P[itrial,2,2]=1-pr
    end
    if PrDw!=PrDw
        error("Nans in PrDw")
    end
    if P!=P
        error("Nans in P")
    end

    if !all(x->x!=Inf,PrDw)
        error("Inf in PrDw")
    end

    return P

end





function Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)

    Pr=ComputePR(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
    Ntrials=length(Pr)
    ll=0

    for itrial in 1:Ntrials
        if choices[itrial]==1
            ll=ll-log(Pr[itrial]+epsilon)
        else
            ll=ll-log(1-Pr[itrial]+epsilon)
        end
    end
    return ll
end


function ModelFitting(stim,delays,idelays,choices,past_choices,past_rewards,args,x,y)

    function LL_f(y)
        x[2]=y[1]
        x[6]=y[2]
        return Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
    end

    Optim.optimize( LL_f,y )

    a=optimize(f, x0, LBFGS(); autodiff = :forward)
    return a




end

function compute_negativ_LL_gradient_hess(stim,delays,idelays,choices,past_choices,past_rewards,args,x)

    function LL_f(x)
        return Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
    end


    grads=ForwardDiff.gradient(LL_f,x)
    return grads
end

function past_choices_rewards(choices,rewards,itrial,n)
    past_choices=zeros(n)
    past_rewards=zeros(n)
    ini=maximum((1,itrial-n))
    fi=itrial-1
    j=1
    for i in fi:-1:ini
        past_choices[j]=choices[i]
        past_rewards[j]=rewards[i]
        j=j+1
    end
    return past_choices,past_rewards
end

function AttractionBias(choices,itrial,mu0a,tau_ab,n)
    past_choices=zeros(n)

    ini=maximum((1,itrial-n))
    fi=itrial-1
    j=1
    for i in fi:-1:ini
        past_choices[j]=choices[i]
        j=j+1
    end
    t=0:(n-1)
    ab=mu0a*sum(past_choices.*exp.(-t/tau_ab) )

    return ab
end




function create_data(Ntrials,delays,T,args,x,consts=0,y=0)

    param=make_dict(args,x,consts,y)
    stim=rand([-1,1],Ntrials)
    MU=[-1,1]

    idelays=StatsBase.sample(1:length(delays),Ntrials)
    n=10 ## number of trials back for the history bias module
    PastChoices=zeros( (Ntrials,n) )
    PastRewards=zeros((Ntrials,n))
    ### PR DW ####
    # PrDw=zeros(2,length(delays))
    # for idelay in 1:length(delays)
    #     coef=[MU[1]*param["mu_k"],param["c2"],param["c4"]]
    #     PrDw[1,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])
    #
    #     coef=[MU[2]*param["mu_k"],param["c2"],param["c4"]]
    #     PrDw[2,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])
    # end

    state=zeros(Ntrials+1)
    choices=zeros(Int8,Ntrials)
    rewards=zeros(Int8,Ntrials)

    state[1]=1
    BiasAttraction=zeros(Ntrials+1)
    println("size stim",size(stim),"size choices",size(choices))

    #println("nslfjbn:" )
    for itrial in 1:Ntrials
        PastChoices[itrial,:],PastRewards[itrial,:]=past_choices_rewards(choices,rewards,itrial,n)

        if state[itrial]==1
            #compute Pr for the DW module
            #Pr=PrDw[stim[itrial],idelays[itrial]]
            #update state
            ab=AttractionBias(choices,itrial,param["mu0_a"],param["tau_a"],n)
            BiasAttraction[itrial]=ab
            coef=[stim[itrial]*param["mu_k"],param["c2"],param["c4"]]
            Pr=PR_1stim(coef,param["sigma"],param["x0"],(param["mu_b"]+ab)*param["mu_k"],delays[idelays[itrial]])


            if rand()<T[1,1]
                state[itrial+1]=1
            else
                state[itrial+1]=2
            end

        else
            #compute Pr for the bias module
            #println(n)

            #Pr=history_bias_module_1stim(param["beta_w"],param["beta_l"],param["tau_w"],param["tau_l"],PastChoices[itrial,:],PastRewards[itrial,:])
            Pr=repeating_module_1stim(param["beta_r"],param["tau_r"],PastChoices[itrial,:])

            #update state
            if rand()<T[2,2]
                state[itrial+1]=2
            else
                state[itrial+1]=1
            end


        end

        if rand()<Pr
            choices[itrial]=1
        else
            choices[itrial]=-1
        end

        if choices[itrial]*sign(stim[itrial])==1
            rewards[itrial]=1
        elseif choices[itrial]*sign(stim[itrial])==-1
            rewards[itrial]=0
        else
            println("something wrong")
        end





    end

    # stim 1,2 to stim -1,1#

    return choices,rewards,state,stim,PastChoices,PastRewards,idelays,BiasAttraction
end





function create_data_history_bias(Ntrials,args,x,consts=0,y=0)

    param=make_dict(args,x,consts,y)
    past_choices=rand(-1:2:1,(Ntrials,10))
    past_rewards=rand(0:1,(Ntrials,10))
    choices=zeros(Ntrials)
    for itrial in 1:Ntrials
        #compute Pr for the bias module
        #Pr=history_bias_module_1stim(param["beta_w"],param["beta_l"],param["tau_w"],param["tau_l"],past_choices[itrial,:],past_rewards[itrial,:])
        Pr=repeating_module_1stim(param["beta_r"],param["tau_r"],past_choices[itrial,:])

        if rand()<Pr
            choices[itrial]=1
        else
            choices[itrial]=-1
        end
    end

    # stim 1,2 to stim -1,1#

    return choices,past_choices,past_rewards
end



function Compute_negative_LL_history_bias(choices,past_choices,past_rewards,args,x,consts=0,y=0)

    param=make_dict(args,x,consts,y)
    Ntrials=(length(past_choices[:,1]) )
    Pr=zeros(typeof(x[1]),Ntrials)
    for itrial in 1:Ntrials
        #Pr[itrial]=history_bias_module_1stim(param["beta_w"],param["beta_l"],param["tau_w"],param["tau_l"],past_choices[itrial,:],past_rewards[itrial,:])
        Pr[itrial]=repeating_module_1stim(param["beta_r"],param["tau_r"],past_choices[itrial,:])

    end
    ll=0
    for itrial in 1:Ntrials
        if choices[itrial]==1
            ll=ll-log(Pr[itrial]+epsilon)
        else
            ll=ll-log(1-Pr[itrial]+epsilon)
        end
    end
    return ll
end







function create_data_WM(Ntrials,delays,args,x,consts=0,y=0)

    param=make_dict(args,x,consts,y)
    stim=rand([1,2],Ntrials)
    MU=[-1,1]
    idelays=StatsBase.sample(1:length(delays),Ntrials)


    ### PR DW ####
    PrDw=zeros(2,length(delays))
    for idelay in 1:length(delays)
        coef=[MU[1]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[1,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])

        coef=[MU[2]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[2,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])
    end

    choices=zeros(Ntrials)

    for itrial in 1:Ntrials
        Pr=0.0
        Pr=PrDw[stim[itrial],idelays[itrial]]
        if rand()<Pr
            choices[itrial]=1
        else
            choices[itrial]=-1
        end
    end

    # stim 1,2 to stim -1,1#

    return choices,stim,idelays
end




function Compute_negative_LL_WM_module(stim,delays,idelays,choices,args,x)

    Pr=ComputePR_WM(stim,delays,idelays,args,x)
    Ntrials=length(Pr)
    ll=0

    for itrial in 1:Ntrials
        if choices[itrial]==1
            ll=ll-log(Pr[itrial]+epsilon)
        else
            ll=ll-log(1-Pr[itrial]+epsilon)
        end
    end
    return ll
end


function ComputePR_WM(stim,delays,idelays,args,x,consts=0,y=0)
    """
    Computes PR for all trials for the wm memory module
    stim: It is an Array with value 1 if stimulus ir R and 2 if stimulus is L. 1 x Ntrials
    delays: Array with all possibles delays in the data
    idelays: Array with the index of the delay corresponding to each trial. 1 x Ntrials
    args: Name of parameters
    x: List of parameters
    """

    Ntrials=length(stim)
    param=make_dict(args,x,consts,y)
    PDw=zeros(typeof(x[1]),Ntrials+1)
    PDw[1]=0.5
    Pr=zeros(typeof(x[1]),Ntrials)
    PrBias=zeros(typeof(x[1]),Ntrials)
    PrDw=zeros(typeof(x[1]),2,length(delays))
    #Compute prob of Right for each combination of stimulus and delay###
    MU=[-1,1]
    for idelay in 1:length(delays)
        coef=[MU[1]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[1,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])


        coef=[MU[2]*param["mu_k"],param["c2"],param["c4"]]
        PrDw[2,idelay]=PR_1stim(coef,param["sigma"],param["x0"],param["mu_b"],delays[idelay])


    end



    for itrial in 1:Ntrials
        Pr[itrial]=PrDw[stim[itrial],idelays[itrial]]
    end

    return Pr

end





function create_data_hmm(PDwDw,PBiasBias,PrDw,PrBias,NTrials)

    state=zeros(typeof(PDwDw),NTrials+1)
    choices=zeros(typeof(PDwDw),NTrials)

    state[1]=1

    for itrial in 1:NTrials
        if state[itrial]==1
            if rand()<PrDw
                choices[itrial]=1
            else
                choices[itrial]=-1
            end
            if rand() <PDwDw
                state[itrial+1]=state[itrial]
            else
                state[itrial+1]=0
            end
        else
            if rand()<PrBias
                choices[itrial]=1
            else
                choices[itrial]=-1
            end
            if rand() <PBiasBias
                state[itrial+1]=state[itrial]
            else
                state[itrial+1]=1
            end
        end
    end
    return choices,state
end

function Compute_negative_LL_hmm_module(PDwDw,PBiasBias,PrDw,PrBias,choices)

    pstate=zeros(typeof(PDwDw),length(choices)+1)
    pstate[1]=0.5
    pr=zeros(typeof(PDwDw),length(choices))
    LL=zeros(typeof(PDwDw),length(choices))
    for itrial in 1:length(choices)
        if itrial==1
            pstate[itrial]=update_PDw(PDwDw,PBiasBias,pstate[itrial],PrDw,PrBias,choices[itrial])
        else
            pstate[itrial]=update_PDw(PDwDw,PBiasBias,pstate[itrial-1],PrDw,PrBias,choices[itrial])
        end
        pr[itrial]=pstate[itrial]*PrDw+(1-pstate[itrial])*PrBias

        if choices[itrial]==1
            LL[itrial]=log(pr[itrial]+epsilon)
        else
            LL[itrial]=log(1-pr[itrial]+epsilon)
        end
    end
    return -sum(LL)
    #return pr,pstate

end



#
# coef=[0.0,1,1]
# roots=zeros(3)
# roots_DW(coef,roots)
# pr=zeros(Float64,1)
# pr[1]=0
# sigma=0.3
# x0=0.0
# initial_transition(coef[1],coef[2],roots[1],roots[3],x0,sigma)
#
# x=-2:0.1:2
# y=zeros(length(x))
# for i in 1:length(x)
#     y[i]=erf(x[i])
# end
# #z=erf.(x)
#
# function error_function(x)
#     return erf(x)
# end
#
#
# x=-2:0.1:2
# y=zeros(length(x))
# y=erf.(x)
# yprima=zeros(length(x))
# for i in 1:length(x)
#     yprima[i]=ForwardDiff.derivative(error_function,x[i])
# end
#
# coef=[0.0,1.0,1.0]
# MU=-0.2:0.01:0.2
# PR=zeros(length(MU))
# x0=0
# DELAYS=[0,100,300,1000]
# figure()
#
# for idelay in 1:length(DELAYS)
#     for i in 1:length(MU)
#         coef[1]=MU[i]
#         PR[i]=PR_1stim(coef,sigma,x0,DELAYS[idelay])
#     end
#     plot(MU,PR,"o-")
#
# end
# show()
