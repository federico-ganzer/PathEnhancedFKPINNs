from pathenhancedfkpinn.problems.base import Parabolic
from pathenhancedfkpinn.problems.heat import Heat
from pathenhancedfkpinn.problems.manufactured import Manufactured
from pathenhancedfkpinn.problems.ou import OU, ou_box

REGISTRY = {"heat": Heat, "manufactured": Manufactured, "ou": OU}


def make_problem(name: str, **kw) -> Parabolic:
    if name not in REGISTRY:
        raise KeyError(f"unknown problem {name!r}; have {sorted(REGISTRY)}")
    if name == "ou" and "d" in kw and "box" not in kw:
        kw["box"] = ou_box(kw["d"])
    if name == "heat" and "d" in kw and "box" not in kw:
        kw["box"] = tuple((-1.0, 1.0) for _ in range(kw["d"]))
    return REGISTRY[name](**kw)


__all__ = ["Parabolic", "Heat", "Manufactured", "OU", "ou_box", "make_problem", "REGISTRY"]
