"""jevcal -- an independent calibration audit for typed-decision models.

Jev's distinguishing claim is not speed or price, both of which are easy to
check. It is that every decision comes with a *calibrated* probability. This
package measures whether that holds, on data with ground-truth labels, and turns
the answer into the one number an engineer actually needs: the confidence
threshold above which it is safe to stop looking.
"""

from .types import Example, Prediction, Task, load_task
from .analysis import AuditResult, analyse
from .runner import align, read_predictions, run_task

__version__ = "0.1.0"

__all__ = [
    "Example",
    "Prediction",
    "Task",
    "load_task",
    "AuditResult",
    "analyse",
    "align",
    "read_predictions",
    "run_task",
    "__version__",
]
