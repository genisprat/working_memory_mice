using PyPlot
using Statistics
using Optim
using ForwardDiff
using JLD
using LineSearches

const epsilon=1e-8
###### definitions ####

Pr1=0.3
Pr2=0.7
P11=0.95
P22=0.96


###### create data ####
Ntrials=1000
state=zeros(Ntrials+1)
choice=zeros(Ntrials)

state[1]=1
for itrial in 1:Ntrials
    if state[itrial]==1
        if rand()<Pr1
            choice[itrial]=1
        else
            choice[itrial]=-1
        end
        if rand()<P11
            state[itrial+1]=1
        else
            state[itrial+1]=2
        end


    elseif state[itrial]==2
        if rand()<Pr2
            choice[itrial]=1
        else
            choice[itrial]=-1
        end
        if rand()<P22
            state[itrial+1]=2
        else
            state[itrial+1]=1
        end
    else
        println("puta")
    end
end

imax=100
figure()
plot(state[1:imax].-1.0,"-k")
plot((choice[1:imax].+1)./2,".k")

index1=findall(x->x==1,state[1:end-1])
index2=findall(x->x==2,state[1:end-1])
p1=mean((choice[index1].+1)./2)
p2=mean((choice[index2].+1)./2)


function ProbabilityState(Pr1,Pr2,P11,P22,choice)

    Ntrials=length(choice)
    PState1=zeros(Ntrials)
    PState2=zeros(Ntrials)


    PState1[1]=0.5
    PState2[1]=0.5
    #println("P11: ",P11," P22 ", P22)
    for itrial in 1:Ntrials
        if itrial==1
            aux1=P11*PState1[itrial]+(1-P22)*PState2[itrial]
            aux2=P22*PState2[itrial]+(1-P11)*PState1[itrial]
        else
            aux1=P11*PState1[itrial-1]+(1-P22)*PState2[itrial-1]
            aux2=P22*PState2[itrial-1]+(1-P11)*PState1[itrial-1]
        end
        #aux1=P11*PState1[itrial-1]+(1-P22)*PState2[itrial-1]
        #aux2=P22*PState2[itrial-1]+(1-P11)*PState1[itrial-1]


        if choice[itrial]==1
            aux12=Pr1*aux1
            aux22=Pr2*aux2
        else
            aux12=(1-Pr1)*aux1
            aux22=(1-Pr2)*aux2
        end
        PState1[itrial]=aux12/(aux12+aux22)
        PState2[itrial]=aux22/(aux12+aux22)
        println("sum Pstates", PState1[itrial]+PState2[itrial])
    end

    return PState1,PState2

end
a,b=ProbabilityState(Pr1,Pr2,P11,P22,choice)
plot(a[1:imax],"r.-")

#
function NegativeLikelihood(Pr1,Pr2,P11,P22,choice)
    Pstate1,Pstate2=ProbabilityState(Pr1,Pr2,P11,P22,choice)
    ll=0.0
    for itrial in 1:length(choice)
        pr=Pstate1[itrial]*Pr1+Pstate2[itrial]*Pr2
        if choice[itrial]==1
            ll=ll-log(pr)
        else
            ll=ll-log(1-pr)
        end
    end

    return ll
end


P1Vector=0.05:0.05:0.95
P2Vector=0.05:0.05:0.95
LLpr=zeros(length(P1Vector),length(P2Vector))
for ip1 in 1:length(P1Vector)
    pr1=P1Vector[ip1]
    for ip2 in 1:length(P2Vector)
        pr2=P2Vector[ip2]
        println(pr1," ",pr2," ",P11," ",P22)
        LLpr[ip1,ip2]=NegativeLikelihood(pr1,pr2,P11,P22,choice)
    end
end


figure()
imshow(LLpr,origin="lower",extent=[P2Vector[1],P2Vector[end],P1Vector[1],P1Vector[end]],aspect="auto",cmap="hot")
xlabel("pr2")
ylabel("pr1")
plot([ Pr1],[Pr2],"bo")
a=findall(x->x==minimum(LLpr),LLpr)
plot([ P2Vector[a[1][2]]],[P1Vector[a[1][1]]],"bs")

colorbar()
show()
