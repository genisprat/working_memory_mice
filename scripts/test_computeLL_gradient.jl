using PyPlot
using Statistics

path_functions="/home/genis/wm_mice/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")

args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.9,     0.8]
param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e4)
choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)

ll=Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)

LL_grads_num=zeros(length(x))
y=zeros(length(x))
dxx=0.00001
for iparam in 1:length(x)
    y=x[:]

    y[iparam]=x[iparam]+dxx
    println(y)
    lly=Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,y)
    LL_grads_num[iparam]=(lly-ll)/dxx

end

println(LL_grads_num)


grads=compute_negativ_LL_gradient_hess(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
#
# function grad_rootsDW(coef)
#
#     function roots2(coef)
#         roots=roots_DW(coef)
#         return roots[1]
#         # roots_aux=zeros(3)
#         # a=sign(-1)
#         # roots[1]=3
#         # roots[2]=a
#         # return 2*coef[1]*roots[1]*(roots_aux[1]+2)*a
#
#     end
#     #roots_aux(coef)
#
#     grad=ForwardDiff.gradient(roots2,coef)
#
#
#
#     return grad
#
#
#
# end
#
#
# roots_DW(coef)
#
# roots=zeros(3)
# coef=[0.2,1.0,1.0]
# a=grad_rootsDW(coef)
# println(a)


#
# function i_t(y)
#
#     return initial_transition(y[1],y[2],-1,1,0,0.3)
# end
#
#
# coef=[0.2,1]
# grad=ForwardDiff.gradient(i_t,coef)
#
# dxx=0.01
# grad_num=zeros(2)
#
# y_coef=i_t(coef)
# for iparam in 1:length(coef)
#     y=coef[:]
#
#     y[iparam]=coef[iparam]+dxx
#     println(y)
#     y_value=i_t(y)
#     grad_num[iparam]=(y_value-y_coef)/dxx
# end



#
# function Transition_rates2(y)
#     println(y)
#     a=zeros(typeof(y[1]),4)
#
#     a[:]=y[:]
#     #println("a:" ,a)
#     b=Transition_rates(a[1:3],-1.0,0.001,1.0,a[4])
#     return 2*b[1]+3*b[2]
# end
# #Transition_rates(coef,xL,xM,xR,sigma)
#
#
# coef=[0.0,1.0,1.0,0.3]
# grad=ForwardDiff.gradient(Transition_rates2,coef)
#
# dxx=0.01
# grad_num=zeros(length(coef))
#
# y_coef=Transition_rates2(coef)
# for iparam in 1:length(coef)
#     y=coef[:]
#
#     y[iparam]=coef[iparam]+dxx
#     #println(y)
#     y_value=Transition_rates2(y)
#     grad_num[iparam]=(y_value-y_coef)/dxx
# end



#
# function Transition_probabilites2(y)
#     println(y)
#     a=zeros(typeof(y[1]),4)
#
#     a[:]=y[:]
#     #println("a:" ,a)
#     b=Transition_probabilites(a[1:3],-1.0,0.001,1.0,a[4],100)
#
#     return 2*b[1]+3*b[2]
# end
# #Transition_rates(coef,xL,xM,xR,sigma)
#
#
# coef=[0.0,1.0,1.0,0.3]
# grad=ForwardDiff.gradient(Transition_probabilites2,coef)
#
# dxx=0.01
# grad_num=zeros(length(coef))
#
# y_coef=Transition_probabilites2(coef)
# for iparam in 1:length(coef)
#     y=coef[:]
#
#     y[iparam]=coef[iparam]+dxx
#     #println(y)
#     y_value=Transition_probabilites2(y)
#     grad_num[iparam]=(y_value-y_coef)/dxx
# end

#
# function PR_1stim2(y)
#     println(y)
#     a=zeros(typeof(y[1]),4)
#
#     a[:]=y[:]
#     #println("a:" ,a)
#     b=PR_1stim(a[1:3],a[4],0.01,0.05,100)
#
#     return 2.0*b
# end
# #Transition_rates(coef,xL,xM,xR,sigma)
#
#
# coef=[0.0,1.0,1.0,0.3]
# grad=ForwardDiff.gradient(PR_1stim2,coef)
#
# dxx=0.01
# grad_num=zeros(length(coef))
#
# y_coef=PR_1stim2(coef)
# for iparam in 1:length(coef)
#     y=coef[:]
#
#     y[iparam]=coef[iparam]+dxx
#     #println(y)
#     y_value=PR_1stim2(y)
#     grad_num[iparam]=(y_value-y_coef)/dxx
# end



#
# function hist_bias2(y)
#     println(y)
#
#     #println("a:" ,a)
#     b=history_bias_module_1stim(y[1],y[2],y[3],y[4],[-1,1,-1,1,1],[1,0,0,1,1])
#
#     #history_bias_module_1stim(beta_w,beta_l,tau_w,tau_l,-1,1)
#
#     return 2.0*b
# end
# #Transition_rates(coef,xL,xM,xR,sigma)
#
#
# coef=[3.0,-1.0,10.0,10.0,]
# grad=ForwardDiff.gradient(hist_bias2,coef)
#
# dxx=0.01
# grad_num=zeros(length(coef))
#
# y_coef=hist_bias2(coef)
# for iparam in 1:length(coef)
#     y=coef[:]
#
#     y[iparam]=coef[iparam]+dxx
#     #println(y)
#     y_value=hist_bias2(y)
#     grad_num[iparam]=(y_value-y_coef)/dxx
# end




function ComputePR2(y)

    #println("a:" ,a)
    Pr=ComputePR(stim,delays,idelays,choices,past_choices,past_rewards,args,y)

    Ntrials=length(Pr)
    ll=0
    #println(Pr)
    for itrial in 1:Ntrials
        if choices[itrial]==1
            ll=ll-log(Pr[itrial]+epsilon)
        else
            ll=ll-log(1-Pr[itrial]+epsilon)
        end
    end
    return ll


    #history_bias_module_1stim(beta_w,beta_l,tau_w,tau_l,-1,1)

    return 2.0*sum(PR)
end
#Transition_rates(coef,xL,xM,xR,sigma)


grad=ForwardDiff.gradient(ComputePR2,x)

grad_num=zeros(length(x))

y_coef=ComputePR2(x)
for iparam in 1:length(x)
    y=x[:]

    y[iparam]=x[iparam]+dxx
    #println(y)
    y_value=ComputePR2(y)
    grad_num[iparam]=(y_value-y_coef)/dxx
end
