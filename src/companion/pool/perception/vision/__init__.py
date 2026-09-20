"""The OpenCV pipeline that reads a pool table from one overhead view.

Ported wholesale from the standalone `htn26` project, and deliberately kept
in its own units: everything here works in millimetres on the measured cloth,
with the origin at the top-left playing corner, x across the width and y down
the length. That is not the contract the planner reads.

The conversion to table-length units - long side 1.0, x along the long side -
happens once, in `perception/service.py`, so that this package stays testable
on its own and there is exactly one place where the axis swap can be wrong.
"""
