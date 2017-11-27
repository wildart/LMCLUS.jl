using Base.Test
using LMCLUS

@testset "LMCLUS Types" begin

    s = LMCLUS.Separation(10., 10., 10., 10, [10.])
    io = IOBuffer()
    serialize(io, s)
    seek(io, 0)
    d = deserialize(io)
    @test s.depth == d.depth
    @test s.discriminability == d.discriminability
    @test s.threshold == d.threshold
    @test s.globalmin == d.globalmin
    @test s.hist_range == d.hist_range
    @test s.threshold == threshold(d)

    m = Manifold(2, [2., 2.], [2. 2.; 2. 2.], [1, 2, 3], 10., 10.)
    @test indim(m) == 2
    @test outdim(m) == 3
    @test labels(m) == [1, 2, 3]
    @test mean(m) == [2., 2.]
    @test projection(m) == [2. 2.; 2. 2.]
    @test threshold(m) == (10., 10.)
    @test assignments([m]) == ones(3)

    io = IOBuffer()
    serialize(io, m)
    seek(io, 0)
    n = deserialize(io)
    @test indim(m) == indim(n)
    @test outdim(m) == outdim(n)
    @test labels(m) == labels(n)
    @test mean(m) == mean(n)
    @test projection(m) == projection(n)
    @test threshold(m) == threshold(n)
end