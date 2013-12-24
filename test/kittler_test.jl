import LMCLUS

xs = rand(1000)
depth, discriminability, threshold, globalmin = kittler(xs)
