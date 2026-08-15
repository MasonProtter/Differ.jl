import DifferentiationInterface as DI
import DifferentiationInterfaceTest as DIT
using Differ

backends = [AutoDifferForwards(), AutoDifferReverse()]
scens = DIT.default_scenarios(; include_constantified = true, include_cachified = true);
DIT.test_differentiation(
    backends, scens;
    excluded = DIT.SECOND_ORDER,
    logging = !parse(Bool, get(ENV, "CI", "false")),
)
