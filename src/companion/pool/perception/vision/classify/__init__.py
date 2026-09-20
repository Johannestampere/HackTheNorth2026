"""A second opinion on what kind each found ball is.

Neither path finds balls; `detector` has already decided where they are and
cut the crops. `net` is the local CNN trained on this table's own balls, and
`baseten` the hosted model kept for comparison.
"""
