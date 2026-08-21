import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using Differ

backends = [
    AutoDifferForwards(),
    AutoDifferReverse(),
]

scens = DIT.default_scenarios(; include_constantified = true, include_cachified = true);

DIT.test_differentiation(
    backends, scens;
    excluded = DIT.SECOND_ORDER,
    logging = !parse(Bool, get(ENV, "CI", "false")),
)

begin
    
    import DifferentiationInterface as DI
    import DifferentiationInterfaceTest as DIT
    using Differ

    backends2 = [
        DI.SecondOrder(AutoDifferForwards(), AutoDifferForwards())
    ]
    
    scens = DIT.default_scenarios(; include_constantified = true, include_cachified = true);
    scens2 = filter(x -> DIT.order(x)==2, scens)
    DIT.test_differentiation(
        backends2, [scens2[1:10]; ];
        logging = !parse(Bool, get(ENV, "CI", "false")),
    )
    
end
