"""
    AutoDifferForwards <: ADTypes.AbstractADType

Selects Differ's forward mode for DifferentiationInterface.jl. Method implementations live in
`DifferForwards.jl`'s own `DifferForwardsDifferentiationInterfaceExt` package extension, loaded
when `DifferentiationInterface` is.
"""
struct AutoDifferForwards <: ADTypes.AbstractADType end

ADTypes.mode(::AutoDifferForwards) = ADTypes.ForwardMode()
