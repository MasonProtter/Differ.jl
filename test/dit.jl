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


backends2 = [
    DI.SecondOrder(AutoDifferForwards(), AutoDifferForwards())
    DI.SecondOrder(AutoDifferForwards(), AutoDifferReverse())
]

scens2 = filter(x -> DIT.order(x)==2, scens)
DIT.test_differentiation(
    backends2, scens2;
    logging = !parse(Bool, get(ENV, "CI", "false")),
)
