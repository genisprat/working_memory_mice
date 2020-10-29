using Distributed


Nconditions=4

rmprocs(workers())
addprocs(4)
@sync @distributed for icondition in 1:Nconditions
    a=rand()
    println(a)
    sleep(10)
    println(a)
end
