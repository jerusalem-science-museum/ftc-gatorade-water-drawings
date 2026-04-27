# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Codebase for the "Drop Screen" exhibit at the Jerusalem Science Museum. A camera captures a visitor, the host PC turns the silhouette into a 64×N binary image, and an Arduino drives a row of solenoid valves to drop water in that shape. Two cooperating codebases live here: a **Python host** (camera + image processing + serial) and **Arduino firmware** (valve / shift-register driver).

## Repo layout — parallel implementations

The least obvious thing about this repo: several folders contain "the same" code at different stages. Edits to one do not propagate.

- [code python/src/](code%20python/src/) — newer, refactored modular Python host. Entry [code python/src/main.py](code%20python/src/main.py); modules `capture.py`, `processing.py`, `display.py`, `arduino.py`, `config.py`. Settings in [code python/src/config.yaml](code%20python/src/config.yaml).
- [code python/currently_running_version/](code%20python/currently_running_version/) — version actually deployed on the exhibit machine. Monolithic [main.py](code%20python/currently_running_version/main.py) with pygame UI, idle-image mode, French comments. Settings in [consts.py](code%20python/currently_running_version/consts.py). Treat as "production"; do not assume it tracks `src/`.
- [code python/camera_version1/](code%20python/camera_version1/) and [code python/camera_version2/](code%20python/camera_version2/) — older snapshots, kept for reference.
- [code python/nathan/](code%20python/nathan/) — collaborator's branch.
- [code arduino/Drop-Screen/](code%20arduino/Drop-Screen/) — Arduino firmware paired with the Python host. Entry `Drop-Screen.ino` plus `Consts.h`, `Routines.h`, `Display.h`.
- [code arduino/Drop-Screen-No-Camera/](code%20arduino/Drop-Screen-No-Camera/) — standalone Arduino sketch that drops hard-coded images, useful for tuning valve timing without a host PC.
- [code arduino/Test-Valves/](code%20arduino/Test-Valves/) and top-level [Test-Valves/](Test-Valves/) — diagnostic sketches for cycling individual valves. Two copies; they are not identical, confirm which one the user means.
- [physics/](physics/) — `simulation.py` and `time_factoring.xlsx` from an abandoned gravity-correction experiment. The deployed firmware deliberately removed this code ("the human eye is correcting the image naturally"). Don't try to revive it without explicit ask.
- [pictures/](pictures/) and [code python/currently_running_version/images_for_idle/](code%20python/currently_running_version/images_for_idle/) — small PNGs shown when no one is in front of the camera.

When the user says "the code", ask which: `src/` (refactor target) or `currently_running_version/` (what's on the exhibit).

## Host ↔ Arduino serial protocol

The load-bearing contract between the two codebases. Spec in [code python/src/arduino API.md](code%20python/src/arduino%20API.md); implementations in [code python/src/arduino.py](code%20python/src/arduino.py) and [code arduino/Drop-Screen/Drop-Screen.ino](code%20arduino/Drop-Screen/Drop-Screen.ino).

- **115200 baud** USB serial. Arduino emits `START\n` after boot.
- **Phase 1 (once per session):** host sends 3 bytes — `image_h`, `valve_on_time_ms`, `drawing_depth`. There is no command to update them later; the Arduino must be reset.
- **Phase 2 (per frame):**
  - `s` (0x73) + 1 cassette byte + `image_h × 8` image bytes + `e` (0x65) → "load new image then drop". Width hard-coded to **64**; image bytes are row-major, MSB = leftmost pixel of the 8, `1` = valve on. The cassette byte (0..`cassettes_num`−1) tells the firmware which physical cassette to drop into. The trailing `e` is a terminator that the firmware reads but does not strictly enforce.
  - `d` (0x64) → drop the buffer that's already loaded.
  - Arduino emits `g` (0x67) every 8 image bytes (flow control, can be drained).
  - Arduino emits `r` (0x72) when drawing is finished. Wait for `r` before the next `s`/`d`. The firmware also emits `Serial.println` text (e.g. `"drawing..."`) which is best drained-and-ignored.
- Pixel packing on the Python side uses `np.packbits(flat, bitorder="big")` — see `_pack_pixels` in [arduino.py](code%20python/src/arduino.py).

## Common commands

### Python host (modular `src/` version)
```bash
pip install -r "code python/requirements.txt"   # pyserial, opencv-contrib-python, numpy, PyYAML
python "code python/src/main.py"
```
Auto-detects the Arduino port; falls back to mock mode (ASCII preview to console) if none found or `arduino_mock: true` in `config.yaml`. Runtime key bindings are printed by `_print_controls` in `main.py` (`r` capture reference background, `s` save config, `m` cycle diff mode, `SPACE` manual send, `q`/ESC quit, etc.).

### Python host (deployed version)
```bash
python "code python/currently_running_version/main.py"
```
Reads `consts.py` and `camera_config.json`. Uses pygame for display, not OpenCV windows.

### Arduino firmware
Open the `.ino` in Arduino IDE — no build script in this repo. Pin wiring is in [code arduino/Drop-Screen/Consts.h](code%20arduino/Drop-Screen/Consts.h) (`SR_*` shift-register pins, RGB LED pins). For valve diagnostics, flash `Test-Valves.ino` instead.

### Tests, lint, build
None. No test suite, no linter config, no CI.

## Things that will bite you

- **Image width is fixed at 64.** Do not change `image_w` on the Arduino or `output_width` in the Python config — the packing assumes 64 px → 8 bytes per row.
- **`output_height` is sent only once per Arduino boot.** Changing it at runtime in the host will desynchronize the buffer until the Arduino is reset.
- **Two `Test-Valves` directories** exist (repo root and under `code arduino/`); confirm which one is meant.
- The two top-level `README.md` files (repo root and `code arduino/`) are duplicates and slightly out of date relative to `code python/src/`.
