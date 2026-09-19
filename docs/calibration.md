# Calibration and physical setup

## What changes with the side-mounted robot

The robot is beside the table, at a distance and roughly 1–2 m high. That height must be clarified as height above floor versus cloth. The RGB camera observes an oblique view; a model estimates depth from that image. Rails and balls can occlude other balls; the far side has less useful image resolution. The projector may cover only part of the table and may struggle to keep the full slanted surface in focus.

Measure the actual table, cameras, projector, and mount before selecting accuracy requirements. Camera field of view and projector coverage are separate constraints.

## Camera calibration: teammate 1

1. Determine RGB lens intrinsics and distortion at the resolution used for inference.
2. Document the RGB depth-estimation model and its output scale. Normalize estimated geometry by the playing surface long side. Output positions and radius in table-length units, not the depth model's raw scale.
3. Determine short-side/long-side ratio, ball radius/long-side ratio, and pocket widths/long-side ratio. Fix the top-left origin in an agreed top-down view, with x right and y down. The long side is 1.0; do not stretch the short side to 1.0.
4. Establish camera pose relative to the table for every repeatable scan view. Measured markers around the rails are a practical starting point when all four corners cannot be seen at once. Marker heights must be accounted for; rail-top markers are not necessarily coplanar with the cloth.
5. Record a calibration ID and resolve each `view_id` to the applicable pose.
6. Evaluate accuracy at near, middle, and far regions against measured positions.

A cloth-plane homography maps planar cloth points; a ball center is above that plane. Directly warping a bounding-box center as if it lay on the cloth creates side-view parallax error. Use calibrated sphere/contour geometry and known radius, with model-estimated depth as supporting evidence. A depth estimate at the visible ball surface is not automatically its center. Intersecting a detected visual-center ray with a center-height plane is an approximation unless the detector geometry supports it; measure its error.

There is no depth sensor or RGB/depth registration step. Evaluate the chosen model on actual oblique table images; validate its normalized geometry against the table proportions. Known real dimensions can supply metric scale from RGB if needed later, but no metric scale is required in perception output.

## Multi-view capture

Start with overlapping, repeatable angles:

1. Blank projected content and confirm a stationary table.
2. Rotate, wait for mechanical/exposure settling, and capture an RGB image.
3. Localize each view in the same table frame.
4. Merge nearby, appearance-compatible detections with uncertainty awareness.
5. Track observed regions and unresolved occlusions separately from ball detections.
6. Reject/restart if balls move during the scan.

This does not inherently require ML for stitching. ML may detect/classify balls, while calibration and geometry combine observations. Rotating a camera from nearly the same optical center expands coverage but does not reliably expose balls hidden behind other objects. Some occlusions require more height or a translated viewpoint. Do not promise complete table state when the mount cannot observe it.

## Projector calibration: teammate 3

For a fixed pose and planar cloth, estimate a homography from normalized table coordinates to output pixels. Use the same top-left origin, x right and y down, with corners `(0,0)`, `(1,0)`, `(0,width)`, and `(1,width)`. This mapping does not require the real table length in meters. Practical first pass:

1. Lock output resolution, display scaling, projector optics, and keystone behavior.
2. Project dots at known pixel locations.
3. Use a calibrated camera to locate the dot hits in table coordinates, or measure them directly.
4. Fit a mapping from table coordinates to projector pixels. At least four well-distributed correspondences are needed; use more points and reserve held-out points for evaluation.
5. Measure held-out dot errors across the useful footprint. Correct or characterize lens distortion when a single homography is insufficient.
6. Save target geometry, homography, resolution, calibration ID, and repeatable pose ID.

The table-to-pixel transform is not the same as the camera-to-table transform. They have different directions and calibrations. The rendering target already stores table-to-pixel H; do not invert it accidentally.

A horizontal motor angle alone does not describe full projector pose. For v1 use discrete calibrated poses. Later, a calibrated kinematic model can derive transforms from motor state, camera/projector extrinsics, and table pose. If only yaw is motorized, pitch must be set mechanically to hit the table.

If a recommended path spans more than the projector footprint, prioritize cue alignment or show regions sequentially. One ordinary projector cannot simultaneously cover disjoint areas by slowly rotating. Test throw distance, focus, brightness against room light, rail shadows, and physical line visibility before committing to whole-table overlays.

## Calibration lifecycle

Moving the table/base, changing camera/projector settings, or altering mounting geometry requires revalidation and potentially a new calibration ID. Rotation repeatability/backlash must be measured; naming a pose does not ensure the hardware reached it. Blank during capture/movement and do not reuse an old frame after balls move.

The scaffold contains **no real calibration artifacts**. All checked-in camera IDs and the projection homography are synthetic. Core metadata checks cannot establish real-world alignment.

Reference for the planar mapping and its limits: [OpenCV homography tutorial](https://docs.opencv.org/4.5.1/d9/dab/tutorial_homography.html). For calibrated perspective geometry: [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/3.4.15/d9/d0c/group__calib3d.html).
