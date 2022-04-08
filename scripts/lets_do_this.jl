using DrWatson
#@quickactivate "Alkali Potentials"
quickactivate("Alkali Potentials")

using PyCall
using PyPlot
using AutoGrad
using LaTeXStrings
np = pyimport("autograd.numpy")
opt = pyimport("scipy.optimize")
mpl = pyimport("mpl_toolkits.axes_grid1")


epsilonh2, sigmah2, c6, c8, c10 = (BigFloat(2.048e-5), BigFloat(7.83), BigFloat(6.49903)
                                                , BigFloat(124.399), BigFloat(3285.83))
Za,Zb, Ea, Eb, Eua, C = (BigFloat(1), BigFloat(1), BigFloat(-0.49973), BigFloat(-0.49973)
                                                    , BigFloat(-2.17503), BigFloat(-1.17556))

function london_coef(c6::BigFloat, c8::BigFloat, c10::BigFloat)
    c = Array{BigFloat,1}(undef, 6)
    c[1] = c6
    c[2] = c8
    c[3] = c10
    for i in 4:1:6
        c[i] = (c[i-1]/c[i-2])^3 * c[i-3]
    end
    return c
end
ch2 = london_coef(c6, c8, c10)

function pot(x, a, b, cn)
    y = BigFloat(0)
    n = 3:1:8
    for i in n
        s = BigFloat(0)
        twoi = BigFloat(2*i)
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s + (b*x).^k/factorial(k)
        end
        f2n = 1-s*exp.(-b*x)
        y = y+ f2n*cn[i-2]/x.^(twoi)
    end
    full = a*exp.(-b*x)-y
    return full
end

function red_pot(x, a, b, cn, sigma, epsilon)
    return pot(x*sigma, a, b, cn)/epsilon
end

a_guess = BigFloat(9.36)
b_guess=BigFloat(1.666)

grad_ = grad(red_pot)
hess_ = grad(grad_)
x_obs = BigFloat(1)
y_obs = BigFloat(-1)


function chi_square(x0)
    a, b = x0
    return (y_obs - red_pot(x_obs, a, b, ch2, sigmah2, epsilonh2))^2
end

function constr1(x0)
    a, b = x0
    return grad_(1., a, b, ch2, sigmah2, epsilonh2)
end

function constr2(x0)
    a, b = x0
    return hess_(1.5, a, b, ch2, sigmah2, epsilonh2)
end
function constr3(x0)
    a, b = x0
    return red_pot(1., a, b, ch2, sigmah2, epsilonh2)+1.
end

tol=BigFloat(1e-10)
nlc1 = opt.NonlinearConstraint(constr1, -tol, +tol)
nlc2 = opt.NonlinearConstraint(constr2, -np.inf, -tol)
nlc3 = opt.NonlinearConstraint(constr3, -tol, +tol)

bnds = ((0, 50), (0, 50))
cons= [nlc1, nlc2, nlc3]

res = opt.minimize(chi_square, (a_guess, b_guess), method="SLSQP", bounds=bnds,
               constraints=cons)

a_fit = BigFloat(res["x"][1])
b_fit = BigFloat(res["x"][2])


beta =1/2*((b_fit-1/sigmah2) + 1/sigmah2*np.sqrt(1+28*sigmah2-2*b_fit*sigmah2+b_fit^2*sigmah2^2))
delta = a_fit*sigmah2^(1-7/beta)*exp(-(b_fit-beta)*sigmah2)


alpha = 1/2*(-(4/sigmah2 + 2*C/(Za*Zb)) + sqrt((4/sigmah2 + 2*C/(Za*Zb))^2-4*(4*C/(Za*Zb*sigmah2)
            -4*epsilonh2/(Za*Zb*sigmah2) + 6/sigmah2^2)))
a1 = C/(Za*Zb) + alpha
a2 = C/(Za*Zb)*alpha + 1/2*alpha^2
a3 = 1/(3*sigmah2^2)*(epsilonh2/(Za*Zb) - a1 - 2*a2*sigmah2)

function short(x, Za, Zb, a1, a2, a3, alpha)::Array{BigFloat}
    return Za*Zb ./ x .* (BigFloat(1.) .+ a1 * x .+ a2 * x.^2. .+ a3 * x.^3.) .* exp.(-alpha *x)
end
betah2 = deepcopy(beta)
function smirnov(x, delta, beta)::Array{BigFloat}
    return delta .* x.^(BigFloat(7) / betah2 - BigFloat(1)) .* exp.(-beta.*x)
end

function dispersion(x, b, cn)::Array{BigFloat}
    y = BigFloat(0.)
    n = 3:1:8
    for i in n
        twoi = 2*i
        twoi = BigFloat(twoi)
        s=BigFloat(0)
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (b .* x) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-b .* x)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    return y
end

function tty(x, delta, beta, b, cn)::Array{BigFloat}
    return smirnov(x, delta, beta) .- dispersion(x, b, cn)
end

function long(x, alpha , delta, beta, b, cn)::Array{BigFloat}
    return (BigFloat(1.) .- exp.(-alpha*x)) .* tty(x, delta, beta, b, cn)
end

function tt2_pot(x, b, cn, delta, beta,  a1, a2, a3, alpha, Za, Zb)::Array{BigFloat}
    return short(x, Za, Zb, a1, a2, a3, alpha) .+ long(x, alpha , delta, beta, b, cn)
end



function tt2_pot2(x, b, cn, delta, beta,  a1, a2, a3, alpha, Za, Zb, sigma, epsilon)::Array{BigFloat}
    a1star = a1*sigma
    a2star = a2*sigma^2
    a3star = a3*sigma^3
    alphastar = alpha*sigma
    betastar = beta*sigma
    bstar = b*sigma
    gammastar = BigFloat(7)/beta - BigFloat(1)
    deltastar = delta/epsilon * sigma^(BigFloat(7)/beta - BigFloat(1))
    cstar = zeros(6)
    xx = x / sigma
    for i in 1:6
        k = i+2
        kk = 2*k
        cstar[i] = cn[i]/(epsilon * sigma^(kk))
    end
    println(cstar)

    short_array = short(xx, Za, Zb, a1star, a2star, a3star, alphastar)
    long_array = long(xx, alphastar , deltastar, betastar, bstar, cstar)

    return 1/(sigma) * short_array .+ (epsilon * long_array)
end





xh2 = collect(range(BigFloat(0.01), BigFloat(15), length=10000))
yh2 = tt2_pot2(xh2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, Za, Zb, sigmah2, epsilonh2)*1e5
#yh2 = tt2_pot(xh2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, Za, Zb) .* BigFloat(1e5)

ylabel="Potential V " * "\$[10^{-5} a.u.]\$"
fig, ax = subplots()
ax.plot(xh2, yh2, label=label=L"$\mathrm{H}_2 \,\mathrm{b}^3 \Sigma_u^+$")
ax.set_yscale("log")
ax.set_xlabel("Distance R [a.u.]")
ax.set_ylabel(ylabel)#
ax.grid()
ax.legend()
#ax.set_ylim(-2.5, 2.5)
gcf()

ind0 = findfirst(x -> x <= 0, yh2) - 1

xlog = xh2[1:ind0]
ylog = yh2[1:ind0]

xneg = xh2[ind0:end]
yneg = yh2[ind0:end]

fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog,label=label=L"$\mathrm{H}_2 \,\mathrm{b}^3 \Sigma_u^+$")
ax.set_yscale("log")

divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg)
axLin.set_xscale("linear")
axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)



data1 = np.array([
#    [6.0, 1-0.999812748],
    [7.0, 1-1.000003776],
    [8.0, 1-1.00002014],
    [9.0, 1-1.000013491],
    [10.0,1-1.000007663],
    [11.0, 1-1.000004320],
    [12.0,1-1.000002516],
    [7.2, 1-1.000012440],
    [7.4, 1-1.000017347],
    [7.6, 1-1.000019730],
    [7.8, 1-1.000020462],
    [7.85, 1-1.000020462],
    [7.9, 1-1.000020405],
    [8.25, 1-1.000018901],
    [8.5, 1-1.000017189],
    [9.5, 1-1.000010230]])

data2 = np.array([
    [1.0, 0.622264306],
    [1.5, 0.809666437],
    [2.0, 0.897076283],
    [2.5, 0.945453719],
    [3.0, 0.972015035],
    [3.5, 0.986130885],
    [4.0, 0.993380059],
    [4.5, 0.996972880],
    [5.0, 0.998687253],
    [5.5, 0.999471748],
    [6.0, 0.999813447],
    [6.5, 0.999952833]
])
data2[:, 2] = 1. .- data2[:, 2]
axLin.scatter(data1[:, 1], data1[:, 2]*1e5, color="C1",label="Kolos 1974")
ax.scatter(data2[:, 1], data2[:, 2]*1e5, color="C3", marker="^", label="Jamieson 2000")
ax.legend()
axLin.legend()
gcf()
savename=plotsdir()*"/h2triplet_zero_inf_potential.png"
savefig(savename)




AUD = BigFloat(5.2917721067e-1)
AUE = BigFloat(219474.6313702)
struct DimerData
    sigma::BigFloat
    eps::BigFloat
    A::BigFloat
    B::BigFloat
    c::BigFloat
    c6::BigFloat
    c8::BigFloat
    c10::BigFloat
    c12::BigFloat
    c14::BigFloat
    c16::BigFloat
end

li2 = DimerData(BigFloat(4.17005)/AUD, BigFloat(333.758)/AUE, BigFloat(0.830339),
 BigFloat(7.310699e-1), BigFloat(2.121980e-3),BigFloat(1.3958e3),BigFloat(0.83546e5),
 BigFloat(0.73828e7),BigFloat(0.96318e9),BigFloat(0.18552e12),
 BigFloat(0.52755e14))


xli2 = collect(range(BigFloat(0.01), BigFloat(2*li2.sigma), length=10000))
xred = collect(range(BigFloat(0.01), BigFloat(2.5), length=10000))
yred_li2 = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, 3, 3)/epsilonh2
yred = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, Za, Zb)/epsilonh2



xli2_pred = xred*li2.sigma
yli2_pred = yred_li2* li2.eps


ind0 = findfirst(x -> x <= 0, yli2_pred) - 1

xlog = xli2_pred[1:ind0]
ylog = yli2_pred[1:ind0]

xneg = xli2_pred[ind0:end]
yneg = yli2_pred[ind0:end]


ylabel="Potential V " * "\$[10^{-3} a.u.]\$"
fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog*1e3,label=label=L"$Li_2 Pred.$")
ax.set_yscale("log")
divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg*1e3)
axLin.set_xscale("linear")
#axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)
gcf()

function pot_ltt(x, a, b, c, cn)
    y = BigFloat(0.)
    n = 3:8
    for i in n
        s=BigFloat(0)
        arg = b .* x .+ 2*c .* x.^2
        twoi = 2*i
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (arg) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-arg)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    full = a .* exp.(-b .* x .- c .* x.^2) .- y
    return full
end


cli2 = [li2.c6, li2.c8, li2.c10, li2.c12, li2.c14, li2.c16]
yli2_ltt = pot_ltt(xli2_pred, li2.A, li2.B, li2.c, cli2)


ind0 = findfirst(x -> x <= 0, yli2_ltt) - 1
xlog = xli2_pred[1:ind0]
ylog = yli2_ltt[1:ind0]

xneg = xli2_pred[ind0:end]
yneg = yli2_ltt[ind0:end]


ax.plot(xlog, ylog*1e3,"--", label=label=L"$Li_2 LTT$")
axLin.plot(xneg, yneg*1e3, "--")
ax.legend()
gcf()
savefig("B:\\owncloud\\toennies\\projekt1\\li2_predicted.png")


#NATRIUM

na2 = DimerData(BigFloat(5.16609)/AUD, BigFloat(173.64960)/AUE, BigFloat(1.129300),
 BigFloat(6.631164e-1), BigFloat(4.974954e-3),BigFloat(1.5616e3),BigFloat(1.1582e5),
 BigFloat(1.1335e7),BigFloat(1.4638e9),BigFloat(0.24944e12),
 BigFloat(0.56089e14))


xna2 = collect(range(BigFloat(0.01), BigFloat(2*na2.sigma), length=10000))
yred_na2 = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, 11, 11)/epsilonh2

xna2_pred = xred*na2.sigma
yna2_pred = yred_na2* na2.eps



ind0 = findfirst(x -> x <= 0, yna2_pred) - 1

xlog = xna2_pred[1:ind0]
ylog = yna2_pred[1:ind0]

xneg = xna2_pred[ind0:end]
yneg = yna2_pred[ind0:end]


ylabel="Potential V " * "\$[10^{-4} a.u.]\$"
fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog*1e4,label=label=L"$Na_2 Pred.$")
ax.set_yscale("log")
divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg*1e4)
axLin.set_xscale("linear")
#axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)
gcf()

function pot_ltt(x, a, b, c, cn)
    y = BigFloat(0.)
    n = 3:8
    for i in n
        s=BigFloat(0)
        arg = b .* x .+ 2*c .* x.^2
        twoi = 2*i
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (arg) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-arg)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    full = a .* exp.(-b .* x .- c .* x.^2) .- y
    return full
end


cna2 = [na2.c6, na2.c8, na2.c10, na2.c12, na2.c14, na2.c16]
yna2_ltt = pot_ltt(xna2_pred, na2.A, na2.B, na2.c, cna2)


ind0 = findfirst(x -> x <= 0, yna2_ltt) - 1
xlog = xna2_pred[1:ind0]
ylog = yna2_ltt[1:ind0]

xneg = xna2_pred[ind0:end]
yneg = yna2_ltt[ind0:end]


ax.plot(xlog, ylog*1e4,"--", label=label=L"$Na_2 LTT$")
axLin.plot(xneg, yneg*1e4, "--")
ax.legend()
gcf()
savefig("B:\\owncloud\\toennies\\projekt1\\na2_predicted.png")



#Kalium

k2 = DimerData(BigFloat(5.7344)/AUD, BigFloat(255.017)/AUE, BigFloat(2.223727),
 BigFloat(6.270931e-1), BigFloat(3.299853e-3),BigFloat(3.9063e3),BigFloat(4.1947e5),
 BigFloat(5.3694e7),BigFloat(8.1929e9),BigFloat(1.4902e12),
 BigFloat(3.2310e14))


xk2 = collect(range(BigFloat(0.01), BigFloat(2*k2.sigma), length=10000))
yred_k2 = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, 19, 19)/epsilonh2

xk2_pred = xred*k2.sigma
yk2_pred = yred_k2* k2.eps



ind0 = findfirst(x -> x <= 0, yk2_pred) - 1

xlog = xk2_pred[1:ind0]
ylog = yk2_pred[1:ind0]

xneg = xk2_pred[ind0:end]
yneg = yk2_pred[ind0:end]


ylabel="Potential V " * "\$[10^{-3} a.u.]\$"
fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog*1e3,label=label=L"$K_2 Pred.$")
ax.set_yscale("log")
divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg*1e3)
axLin.set_xscale("linear")
#axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)
gcf()

function pot_ltt(x, a, b, c, cn)
    y = BigFloat(0.)
    n = 3:8
    for i in n
        s=BigFloat(0)
        arg = b .* x .+ 2*c .* x.^2
        twoi = 2*i
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (arg) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-arg)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    full = a .* exp.(-b .* x .- c .* x.^2) .- y
    return full
end


ck2 = [k2.c6, k2.c8, k2.c10, k2.c12, k2.c14, k2.c16]
yk2_ltt = pot_ltt(xk2_pred, k2.A, k2.B, k2.c, ck2)


ind0 = findfirst(x -> x <= 0, yk2_ltt) - 1
xlog = xk2_pred[1:ind0]
ylog = yk2_ltt[1:ind0]

xneg = xk2_pred[ind0:end]
yneg = yk2_ltt[ind0:end]


ax.plot(xlog, ylog*1e3,"--", label=label=L"$K_2 LTT$")
axLin.plot(xneg, yneg*1e3, "--")
ax.legend()
gcf()
savefig("B:\\owncloud\\toennies\\projekt1\\k2_predicted.png")


#Rubidium

rb2 = DimerData(BigFloat(6.065)/AUD, BigFloat(241.5045)/AUE, BigFloat(2.371131),
 BigFloat(5.953130e-1), BigFloat(3.865031e-3),BigFloat(4.6669e3),BigFloat(5.7202e5),
 BigFloat(7.9366e7),BigFloat(12.465e9),BigFloat(2.2162e12),
 BigFloat(4.4601e14))


xrb2 = collect(range(BigFloat(0.01), BigFloat(2*rb2.sigma), length=10000))
yred_rb2 = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, 37, 37)/epsilonh2

xrb2_pred = xred*rb2.sigma
yrb2_pred = yred_rb2* rb2.eps



ind0 = findfirst(x -> x <= 0, yrb2_pred) - 1

xlog = xrb2_pred[1:ind0]
ylog = yrb2_pred[1:ind0]

xneg = xrb2_pred[ind0:end]
yneg = yrb2_pred[ind0:end]


ylabel="Potential V " * "\$[10^{-3} a.u.]\$"
fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog*1e3,label=label=L"$Rb_2 Pred.$")
ax.set_yscale("log")
divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg*1e3)
axLin.set_xscale("linear")
#axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)
gcf()

function pot_ltt(x, a, b, c, cn)
    y = BigFloat(0.)
    n = 3:8
    for i in n
        s=BigFloat(0)
        arg = b .* x .+ 2*c .* x.^2
        twoi = 2*i
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (arg) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-arg)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    full = a .* exp.(-b .* x .- c .* x.^2) .- y
    return full
end


crb2 = [rb2.c6, rb2.c8, rb2.c10, rb2.c12, rb2.c14, rb2.c16]
yrb2_ltt = pot_ltt(xrb2_pred, rb2.A, rb2.B, rb2.c, crb2)


ind0 = findfirst(x -> x <= 0, yrb2_ltt) - 1
xlog = xrb2_pred[1:ind0]
ylog = yrb2_ltt[1:ind0]

xneg = xrb2_pred[ind0:end]
yneg = yrb2_ltt[ind0:end]


ax.plot(xlog, ylog*1e3,"--", label=label=L"$Rb_2 LTT$")
axLin.plot(xneg, yneg*1e3, "--")
ax.legend()
gcf()
savefig("B:\\owncloud\\toennies\\projekt1\\rb2_predicted.png")




#Caesium

cs2 = DimerData(BigFloat(6.3055)/AUD, BigFloat(278.581)/AUE, BigFloat(3.748305),
 BigFloat(6.099663e-1), BigFloat(2.010526e-3),BigFloat(6.7328e3),BigFloat(10.032e5),
 BigFloat(15.792e7),BigFloat(26.263e9),BigFloat(4.6143e12),
 BigFloat(8.5651e14))


xcs2 = collect(range(BigFloat(0.01), BigFloat(2*cs2.sigma), length=10000))
yred_cs2 = tt2_pot(xred*sigmah2, b_fit, ch2, delta, beta,  a1, a2, a3, alpha, 1, 1)/epsilonh2

xcs2_pred = xred*cs2.sigma
ycs2_pred = yred_cs2* cs2.eps



ind0 = findfirst(x -> x <= 0, ycs2_pred) - 1

xlog = xcs2_pred[1:ind0]
ylog = ycs2_pred[1:ind0]

xneg = xcs2_pred[ind0:end]
yneg = ycs2_pred[ind0:end]


ylabel="Potential V " * "\$[10^{-3} a.u.]\$"
fig, ax = plt.subplots(dpi=600)
ax.plot(xlog, ylog*1e3,label=label=L"$Cs_2 Pred.$")
ax.set_yscale("log")
divider = mpl.make_axes_locatable(ax)
axLin = divider.append_axes("bottom", size=2.0, pad=0.02, sharex=ax)
axLin.plot(xneg, yneg*1e3)
axLin.set_xscale("linear")
#axLin.set_ylim((-3, Float32(yh2[ind0])))
axLin.spines["top"].set_visible(0)
ax.spines["bottom"].set_visible(0)
ax.xaxis.set_ticks_position("top")
ax.grid()
axLin.set_xlabel("Distance R [a.u.]")
axLin.grid()
ax.set_ylabel(ylabel)
gcf()

function pot_ltt(x, a, b, c, cn)
    y = BigFloat(0.)
    n = 3:8
    for i in n
        s=BigFloat(0)
        arg = b .* x .+ 2*c .* x.^2
        twoi = 2*i
        for k in 0:1:twoi
            k = BigFloat(k)
            s = s .+ (arg) .^k ./ factorial(k)
        end
        f2n = BigFloat(1) .- s .* exp.(-arg)
        y = y .+ f2n .* cn[i-2] ./ x.^(twoi)
    end
    full = a .* exp.(-b .* x .- c .* x.^2) .- y
    return full
end


ccs2 = [cs2.c6, cs2.c8, cs2.c10, cs2.c12, cs2.c14, cs2.c16]
ycs2_ltt = pot_ltt(xcs2_pred, cs2.A, cs2.B, cs2.c, ccs2)


ind0 = findfirst(x -> x <= 0, ycs2_ltt) - 1
xlog = xcs2_pred[1:ind0]
ylog = ycs2_ltt[1:ind0]

xneg = xcs2_pred[ind0:end]
yneg = ycs2_ltt[ind0:end]


ax.plot(xlog, ylog*1e3,"--", label=label=L"$Cs_2 LTT$")
axLin.plot(xneg, yneg*1e3, "--")
ax.legend()
gcf()
savefig("B:\\owncloud\\toennies\\projekt1\\cs2_predicted.png")