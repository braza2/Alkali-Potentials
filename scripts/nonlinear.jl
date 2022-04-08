using DrWatson
#@quickactivate "Alkali Potentials"
quickactivate("Alkali Potentials")

using PyCall
using PyPlot
using AutoGrad
using LaTeXStrings
using NLsolve

# x[1]=a1
# x[2]=a2
# x[3]=a3
# x[4]=α
# x[5]=δ


function f!(F, x)
    F[1]=