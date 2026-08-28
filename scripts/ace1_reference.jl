"""
Evaluate an ACE fixture directly through ACE1.jl.

The first argument is a generated Julia input file defining the uppercase
constants used below.  The second argument receives a plain-text matrix, one
atom per row.  Keeping the transport format in Julia's standard library avoids
adding a JSON or NumPy package to the archived ACE1.jl project.
"""

using ACE1
using JuLIP

include(ARGS[1])

basis = ACE1.Utils.rpi_basis(
    species=SPECIES,
    N=N_ORDER,
    r0=R0,
    trans=ACE1.PolyTransform(TRANSFORM_P, TRANSFORM_R0),
    wL=WL,
    maxdeg=MAX_DEGREE,
    D=DEGREES,
    rcut=R_CUT,
    rin=R_IN,
    pcut=P_CUT,
    pin=P_IN,
    constants=CONSTANTS,
    warn=false,
)

blocks = Matrix{Float64}[]
for structure in 1:(length(OFFSETS) - 1)
    first = OFFSETS[structure]
    last = OFFSETS[structure + 1] - 1
    atoms = JuLIP.Atoms(
        NUMBERS[first:last],
        permutedims(POSITIONS[first:last, :]);
        cell=CELLS[structure],
        pbc=Tuple(PBCS[structure, :]),
    )
    push!(blocks, Matrix(permutedims(ACE1.Descriptors.descriptors(basis, atoms))))
end
values = isempty(blocks) ? zeros(Float64, 0, 0) : vcat(blocks...)

open(ARGS[2], "w") do output
    println(output, size(values, 1), " ", size(values, 2))
    for row in axes(values, 1)
        println(output, join((repr(values[row, column]) for column in axes(values, 2)), " "))
    end
end

println("ACE1=", Base.pkgversion(ACE1))
println("JuLIP=", Base.pkgversion(JuLIP))
println("Julia=", VERSION)

