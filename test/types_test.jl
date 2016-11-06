using Base.Test
using LMCLUS

@testset "LMCLUS Types" begin

    s = Separation(10., 10., 10., 10, [10.])
    io = IOBuffer()
    serialize(io, s)
    seek(io, 0)
    d = deserialize(io)
    @test s.depth == d.depth
    @test s.discriminability == d.discriminability
    @test s.threshold == d.threshold
    @test s.globalmin == d.globalmin
    @test s.hist_range == d.hist_range
    @test s.threshold == threshold(s)

    m = Manifold(2, [2., 2.], [2. 2.; 2. 2.], [1, 2, 3], s)
    @test indim(m) == 2
    @test outdim(m) == 3
    @test labels(m) == [1, 2, 3]
    @test mean(m) == [2., 2.]
    @test projection(m) == [2. 2.; 2. 2.]
    @test separation(m) == s
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

    S = separation(n)
    @test s.depth == S.depth
    @test s.discriminability == S.discriminability
    @test s.threshold == S.threshold
    @test s.globalmin == S.globalmin
    @test s.hist_range == S.hist_range
    @test s.threshold == threshold(S)
end