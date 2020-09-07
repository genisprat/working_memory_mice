using PyPlot
using Statistics
using Pandas
path_functions="/home/genis/wm_mice/"
path_figures="/home/genis/wm_mice/figures/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")

args=["mu_k","c2","c4","x0","mu_b","sigma","beta_w","beta_l","tau_w","tau_l","PDwDw","PBiasBias"]
x=[    0.3,  1.2, 1.0, 0.15, -0.05,   0.3,      3.0,     -1.0,     10,     10,    0.95,     0.8]
param=make_dict2(args,x)
delays=[0.0,100,200,300,500,800,1000]
Ntrials=Int(1e4)
choices,state,stim,past_choices,past_rewards,idelays=create_data(Ntrials,delays,args,x)
#change stim from 1,2 to -1,1
dict=Dict(:choices=>choices,:stim=>stim,:past_choices=>past_choices,:past_rewards=>past_rewards,:idelays=>idelays)
#dict=Dict(:choices=>choices,:stim=>stim,:idelays=>idelays)
PAST_CHOICES=[]
PAST_REWARDS=[]
for itrial in 1:Ntrials
    push!(PAST_CHOICES,past_choices[itrial,:])
    push!(PAST_REWARDS,past_rewards[itrial,:])
end
dict=Dict(:choices=>choices,:stim=>stim,:past_choices=>PAST_CHOICES,:past_rewards=>PAST_REWARDS,:idelays=>idelays)

df=Pandas.DataFrame(dict)

#
filename_save="/home/genis/wm_mice/synthetic_data/synthetic_data_to_python.json"
Pandas.to_json(df,filename_save)
