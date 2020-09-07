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



C2=0.5:0.1:2.
SIGMA=0.05:0.05:0.6


LL1=zeros(length(C2),length(SIGMA))
for ic2 in 1:length(C2)
    x[2]=C2[ic2]
    for isigma in 1:length(SIGMA)
        x[6]=SIGMA[isigma]
        LL1[ic2,isigma]=Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
    end
end
y=x[:]
x[2]=1.3
x[6]=0.33
ll_special=Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
x[2]=param["c2"]
x[6]=param["sigma"]

figure()
imshow(LL1,origin="lower",extent=[SIGMA[1],SIGMA[end],C2[1],C2[end]],aspect="auto",cmap="hot")
xlabel("sigma")
ylabel("c2")
plot([param["sigma"]], [ param["c2"]],"bo")
colorbar()
show()

figure()
imshow(LL1,origin="lower",extent=[SIGMA[1],SIGMA[end],C2[1],C2[end]],aspect="auto",cmap="hot",vmax=6000)
xlabel("sigma")
ylabel("c2")
plot([param["sigma"]], [ param["c2"]],"bo")
colorbar()
show()





############## beta #########
# BETA_W=-5:0.5:5
#
# BETA_L=-5:0.5:3
#
#
# LL2=zeros(length(BETA_W),length(BETA_L))
# for ibetaw in 1:length(BETA_W)
#     x[7]=BETA_W[ibetaw]
#     for ibetal in 1:length(BETA_L)
#         x[8]=BETA_L[ibetal]
#         LL2[ibetaw,ibetal]=Compute_negative_LL(stim,delays,idelays,choices,past_choices,past_rewards,args,x)
#     end
# end
#
#
# figure()
# imshow(LL2,origin="lower",extent=[BETA_L[1],BETA_L[end],BETA_W[1],BETA_W[end]],aspect="auto",cmap="hot")
# xlabel("beta_l")
# ylabel("beta_w")
# plot([param["beta_l"]], [ param["beta_w"]],"bo")
# colorbar()
# show()
#
# figure()
# imshow(LL2,origin="lower",extent=[BETA_L[1],BETA_L[end],BETA_W[1],BETA_W[end]],aspect="auto",cmap="hot",vmax=6000)
# xlabel("beta_l")
# ylabel("beta_w")
# plot([param["beta_l"]], [ param["beta_w"]],"bo")
# colorbar()
# show()
