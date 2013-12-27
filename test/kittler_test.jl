import LMCLUS
using Base.Test

xs = rand(1000)
depth, discriminability, threshold, globalmin = kittler(xs)
