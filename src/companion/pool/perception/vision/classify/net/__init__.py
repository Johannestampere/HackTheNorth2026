"""A learned second opinion on what each ball is.

`detect.balls` decides the four kinds from measurements: how much white sits
in a thin annulus at the ball's rim, how dark and colourless its body is. Those
measurements have a ceiling that is a property of one overhead view rather than
of any threshold. A stripe lying with its coloured band square to the camera
shows no white at all and measures exactly like a solid; a solid with its
number facing up measures like a stripe. Held out over layouts the model had
never seen, the rim test called 62.8% of the crops in `crops/` correctly.

The same crops, shown to a small CNN: 86.7%.

That gap is not a better threshold waiting to be found. It is the difference
between asking "how much white is near the edge" and asking "does this look
like a stripe", and only the second question survives a ball being rotated.

What is deliberately *not* here
-------------------------------
The model does not find balls, and it does not replace the geometry. It is
handed a crop that the detector already decided to look at, cut at the same
1.7 r the training crops were cut at, and it answers with one of five labels.
Everything that makes the reading trustworthy - the table measurement, the
known radius, one cue and one 8 per set, seven stripes and seven solids - is
arithmetic in `detect.balls` and stays there. A network that is merely usually
right must not be allowed to overrule a fact about the game.

Why a CNN rather than the hosted model
--------------------------------------
Measured on this table: one Baseten round trip is 0.54-0.83 s and the whole
table took 57-78 s, against 0.38 s for all of OpenCV, and it scored 13/16
where OpenCV scored 14/16. This runs in about 6 ms for all sixteen crops on
the CPU that is already here, and it is trained on this table's own balls.
"""

from __future__ import annotations

from .kinds import (LABELS, BallNet, crop_size, load_classifier,
                    model_path, predict)

__all__ = ["LABELS", "BallNet", "crop_size", "load_classifier", "model_path",
           "predict"]
