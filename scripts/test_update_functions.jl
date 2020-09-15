using PyPlot
using Statistics

path_functions="/home/genis/wm_mice/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")



############### Test update function ##############
Ntrials=50
p=[0.8,0.2]
pstay=[0.95,0.2]
state=zeros(Int,Ntrials+1)
pstate=zeros(Ntrials+1)
y=zeros(Ntrials)
state[1]=2
pstate[1]=0.5
for itrial in 1:Ntrials
     if rand()<p[state[itrial]]
         y[itrial]=1
    else
         y[itrial]=-1
    end
    if rand()<pstay[state[itrial]]
        state[itrial+1]=state[itrial]
    else
        if state[itrial]==1
            state[itrial+1]=2
        else
            state[itrial+1]=1
        end
    end
    pstate[itrial+1]=update_PDw(pstay[1],pstay[2],pstate[itrial],p[1],p[2],y[itrial])
end

figure()
plot(y,"o")
plot(state,"b-")
plot(pstate,"r-")
show()
