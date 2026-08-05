"""
    AutoDifferForwards <: ADTypes.AbstractADType

Selects Differ's forward mode for DifferentiationInterface.jl. Method implementations live in the
`DifferDifferentiationInterfaceExt` package extension, loaded when `DifferentiationInterface` is.
"""
struct AutoDifferForwards <: ADTypes.AbstractADType end

"""
    AutoDifferReverse <: ADTypes.AbstractADType

Selects Differ's reverse mode for DifferentiationInterface.jl. Method implementations live in the
`DifferDifferentiationInterfaceExt` package extension, loaded when `DifferentiationInterface` is.
"""
struct AutoDifferReverse <: ADTypes.AbstractADType end

ADTypes.mode(::AutoDifferForwards) = ADTypes.ForwardMode()
ADTypes.mode(::AutoDifferReverse) = ADTypes.ReverseMode()
