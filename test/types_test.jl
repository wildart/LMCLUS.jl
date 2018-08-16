using LMCLUS
using Test
using Serialization
using Statistics

@testset "LMCLUS Types" begin

    s = LMCLUS.Separation(10., 10., 10., 10, 10.0, 10.0, 10)
    io = IOBuffer()
    serialize(io, s)
    seek(io, 0)
    d = deserialize(io)
    @test s.depth == d.depth
    @test s.discriminability == d.discriminability
    @test s.threshold == d.threshold
    @test s.globalmin == d.globalmin
    @test s.threshold == threshold(s)
    @test s.maxdist == d.maxdist
    @test s.mindist == d.mindist
    @test s.bins == d.bins

    m = Manifold(2, [2., 2.], [2. 2.; 2. 2.], [1, 2, 3], 10., 10.)
    @test outdim(m) == 2
    @test size(m) == 3
    @test points(m) == [1, 2, 3]
    @test mean(m) == [2., 2.]
    @test projection(m) == [2. 2.; 2. 2.]
    @test threshold(m) == (10., 10.)

    io = IOBuffer()
    serialize(io, m)
    seek(io, 0)
    n = deserialize(io)
    @test outdim(m) == outdim(n)
    @test size(m) == size(n)
    @test points(m) == points(n)
    @test mean(m) == mean(n)
    @test projection(m) == projection(n)
    @test threshold(m) == threshold(n)

end
