my_tests = ["kittler_test.jl", "lmclus_test.jl", "lmclus_p_test.jl", "mdl_test.jl"]

println("Running tests:")

for my_test in my_tests
    println(" * $(my_test)")
    include(my_test)
end