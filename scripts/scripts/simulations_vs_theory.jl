using PyPlot
using Statistics

path_functions="/home/genis/wm_mice/"

include(path_functions*"functions_wm_mice.jl")
include(path_functions*"function_simulations.jl")




tau=1
T=1000
dt=tau/40
NT=Int(T/dt)
dt_sqrt=sqrt(dt)
x0=0
sigma=0.25
c2=1.0
c4=1.0
MU=-0.3:0.05:0.3
Ntrials=1000
coef=[0.0,c2,c4]

DELAYS=[0,100,500,1000]

PR_sim=zeros(length(DELAYS),length(MU))
PR=zeros(length(DELAYS),length(MU))


for idelay in 1:length(DELAYS)
    T=DELAYS[idelay]
    NT=Int(T/dt)
    println("idelay: ", idelay)

    for imu in 1:length(MU)
        println("imu: ", imu)
        coef[1]=MU[imu]
        internal_noise=sigma.*randn(Ntrials,NT)
        NT2=Int(10*tau/dt)
        internal_noise2=sigma.*randn(Ntrials,NT2)
        d=simulation_DW_WM(coef,x0,internal_noise,internal_noise2,dt)
        PR_sim[idelay,imu]=mean((d.+1)./2)

        PR[idelay,imu]=PR_1stim(coef,sigma,x0,T)

    end

end

colors=["red","blue","orange","green"]
figure()
for idelay in 1:length(DELAYS)
    plot(MU,PR[idelay,:],color=colors[idelay])
    plot(MU,PR_sim[idelay,:],"o",color=colors[idelay],label="D="*string(DELAYS[idelay]))
end
xlabel("MU")
ylabel("PR")
legend()
show()
