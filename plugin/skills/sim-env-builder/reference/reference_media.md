# Recreating a scene from the user's photos or videos

The user gives media ("I saved two videos in ./reference/", a photo of their lab) and wants the
environment to look like it. The media is the spec: read it before the brief, build against it,
and prove the match with a side-by-side image.

## 1. Extract and look (before the brief)

```bash
$PY -m simkit.reference extract reference/ --out <env>/reference/frames --per-video 8
```
LOOK at `<env>/reference/sheet.jpg` and 2-4 single frames at full size. Write down:
- **Layout**, as a top-down sketch in metres: which streets/rooms/zones, which side each is on as
  seen from the camera (left/right/near/far), where the camera stands.
- **Objects with counts and sizes**, scaled from things of known size: car 4.5-4.9 m long and
  1.8-1.9 m wide, lane 3.0-3.6 m, kerb 0.15 m, door 2.0-2.1 m, storey 3.0-3.5 m (Victorian
  bay-window houses ~3.5-4 m), traffic signal head ~1 m on a 3-5 m pole, mature park tree 15-30 m.
- **Viewpoint**: camera height (ground floor 1.6 m, each storey +3-3.5 m), position, what it looks
  at, orientation (portrait video = tall frame), field of view (phone main camera ~65-70 deg along
  the long side; 0.5x ultrawide ~110 deg).
- **Dynamics shown**: what moves and how (water flowing toward the camera, fountains pulsing,
  cars queueing then crossing), weather and light (overcast, rain, fog in the distance, wet road).
- **What cannot be reproduced** in MuJoCo (rain drops, splashing spray, photoreal foam, logos)
  and the stand-in you will use. Never copy logos, livery or brand names from the media.

## 2. Brief

Add a "From your media" block to the brief: the layout sketch in words (5-8 lines), the object
list with counts, the camera vantage you will reproduce, and the approximations. Ask only what the
media leaves open (e.g. "the far side is out of frame: park continues [default] or buildings?").
Copy the layout and viewpoint into SPEC.md under "## Reference" with the frame names you used.

## 3. Build against it

- A named camera `cam_reference` at the media's vantage point, plus one per distinct viewpoint if
  the media has several.
- Write `<env>/reference/match.json`; the smoke test then renders `smoke/reference_compare.png`
  (reference | your camera | 50 % overlay) every run:
  `[{"camera": "cam_reference", "frame": "reference/frames/<best frame>.jpg", "seconds": 0}]`
  Use `"seconds"` to move the demo to the moment the frame shows (a car at the corner).
- Iterate the camera first (pos, target, fovy) until horizon, kerb lines and the big landmarks line
  up in the overlay; then fix the layout (positions, sizes, counts) until the big objects line up.
  2-3 rounds is normal. Do not tune by eye on the contact sheet alone.
- Colours: sample them from the frames (water, road, grass, building paint), not from defaults.

## 4. Check and report

Visual check item 10 (SMOKE.md): compare `reference_compare.png` with the media and list every
visible mismatch. The report says what matches, what was approximated, and what is missing.
