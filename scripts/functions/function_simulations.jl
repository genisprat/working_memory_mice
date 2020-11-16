using PyPlot
using Statistics

path_functions="/home/genis/wm_mice/scripts/functions/"

include(path_functions*"functions_wm_mice.jl")

function simulation_DW_WM(coef,x0,internal_noise,internal_noise2,dt)

    Ntriasl,NT=size(internal_noise)
    Ntrials,NT2=size(internal_noise2)
    d=zeros(Ntrials)
    dt_sqrt=sqrt(dt)
    roots=zeros(3)
    roots_DW(coef,roots)
    for itrial in 1:Ntrials
        x=x0
        it=1
        while((x<0.9*roots[3] && x>0.9*roots[1]) && it <NT2 )
            x=x-(-coef[1]-coef[2]*x+coef[3]*x^3)*dt+dt_sqrt*internal_noise2[itrial,it]
            it=it+1
        end
        #println(it)
        for it in 1:NT
            x=x-(-coef[2]*x+coef[3]*x^3)*dt+dt_sqrt*internal_noise[itrial,it]
        end
        d[itrial]=sign(x)
    end

    return d
end
