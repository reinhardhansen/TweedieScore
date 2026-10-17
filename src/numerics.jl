# Special functions and small numerical helpers.
#
# The Python code takes expit, logsumexp, gammaln and wrightomega from SciPy and
# logaddexp and linspace from NumPy. They are reimplemented here so that the
# package needs no dependency beyond the standard list in Project.toml.

"""
    expit(x)

Logistic function 1/(1 + exp(-x)), evaluated without overflow for either sign
of `x` (scipy.special.expit).
"""
expit(x::Real) = x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x))

"""
    log1pexp(x)

log(1 + exp(x)), evaluated as in numpy.logaddexp(0, x).
"""
log1pexp(x::Real) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))

"""
    logsumexp(v)

log(sum(exp.(v))) with the maximum subtracted first (scipy.special.logsumexp).
"""
function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    isfinite(m) || (m = 0.0)
    s = 0.0
    @inbounds for x in v
        s += exp(x - m)
    end
    return m + log(s)
end

"""
    wrightomega(x)

Wright omega function for real `x`: the solution w > 0 of w + log(w) = x
(scipy.special.wrightomega). For real arguments w = W0(exp(x)), but the
Lambert form overflows for large `x`, so the equation is solved directly.

With u = log(w) the equation is g(u) = exp(u) + u - x = 0. Since g is convex
and increasing, Newton's method converges monotonically from the first iterate
on, for every starting value. One Newton step on w + log(w) = x polishes the
result. In a comparison over the arguments used in the risk experiment the
result agrees with SciPy to within one unit in the last place of w.
"""
function wrightomega(x::Real)
    x = Float64(x)
    isnan(x) && return NaN
    x == Inf && return Inf
    x == -Inf && return 0.0
    u = x <= 1.0 ? x : log(x)
    for _ in 1:100
        eu = exp(u)
        step = (eu + u - x) / (eu + 1.0)
        u -= step
        abs(step) <= 1.0e-15 * max(1.0, abs(u)) && break
    end
    w = exp(u)
    w > 0.0 || return w
    return w - (w + log(w) - x) / (1.0 + 1.0 / w)
end

"""
    logfactorials(n) -> Vector{Float64}

Vector `lf` of length n + 1 with lf[k + 1] = log(k!) for k = 0, ..., n, formed
by compensated (Neumaier) summation of log(1), ..., log(n). This replaces
scipy.special.gammaln(y + 1) for the Poisson counts; the two agree to within a
few units in the last place.
"""
function logfactorials(n::Integer)
    out = zeros(Float64, n + 1)
    s = 0.0
    c = 0.0
    for k in 1:n
        v = log(Float64(k))
        t = s + v
        if abs(s) >= abs(v)
            c += (s - t) + v
        else
            c += (v - t) + s
        end
        s = t
        out[k + 1] = s + c
    end
    return out
end

"""
    numpy_linspace(start, stop, num)

Equally spaced points computed as numpy.linspace does: start + i * step for
i = 0, ..., num - 1, with the last point set to `stop`.
"""
function numpy_linspace(start::Real, stop::Real, num::Integer)
    step = (stop - start) / (num - 1)
    out = [i * step + start for i in 0:(num - 1)]
    out[end] = stop
    return out
end

"""
    fmt_g(x)

Format a number as Python's f"{x:g}" (for example 3.0 -> "3", 0.25 -> "0.25").
"""
fmt_g(x::Real) = @sprintf("%g", x)

"""
    child_seeds(seed, n)

Integer seeds for `n` independent streams derived from one master seed. This
plays the role of numpy's SeedSequence(seed).spawn(n); the streams themselves
differ from NumPy's.
"""
child_seeds(seed::Integer, n::Integer) = rand(Random.Xoshiro(seed), UInt64, n)
