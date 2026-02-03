# Drop-Screen Arduino – Host API

Serial protocol for the Raspberry Pi (or other host) talking to the Drop-Screen Arduino.

---

## Connection

| Setting | Value |
|--------|--------|
| Baud rate | **115200** |
| Port | Serial (UART) to Arduino |

After boot, the Arduino sends `"START"` + newline. You can wait for this before sending.

---

## Phase 1: Parameters (once per session)

Send **exactly 3 bytes** in this order **before** any image commands. These are read only once per power-up; there is no command to reset or update them.

| # | Byte | Meaning | Type | Example |
|---|------|---------|------|--------|
| 0 | param0 | `image_h` | Image height (rows) | 40 |
| 1 | param1 | `valve_on_time` | ms each row’s valves are on | 5 |
| 2 | param2 | `drawing_depth` | Layers per image | 1 |

- **image_h** – Height in rows (e.g. 40). Width is fixed at **64** on the Arduino.
- **valve_on_time** – Milliseconds each row is displayed (e.g. 5).
- **drawing_depth** – Number of “layers” per image (e.g. 1).

---

## Phase 2: Commands

Send only when the Arduino is **not** drawing. Wait for **`r`** (ready) before sending the next command.

### Drop current buffer — `d`

- Send one byte: **`d`** (0x64).
- Arduino runs a drawing with whatever is already in the image buffer (e.g. from a previous **Send new image**).

### Send new image then drop — `s`

- Send one byte: **`s`** (0x73).
- Then send **exactly** `image_h × 8` bytes of image data (e.g. 320 bytes for height 40).
- Arduino may send **`g`** after every **8** image bytes (flow control).
- After the last byte, Arduino waits 10 ms then starts the drawing.

---

## Arduino → Host

| Byte | When |
|------|------|
| **`g`** (0x67) | After every 8 image bytes during a **Send new image** (`s`) sequence. |
| **`r`** (0x72) | When the current drawing is finished. Safe to send the next command. |

Wait for **`r`** before sending the next `s` or `d`.

---

## Image format (for `s` command)

- **Layout:** Row-major. Row 0 first, then row 1, …, then row `image_h - 1`.
- **Row size:** 8 bytes per row (64 pixels ÷ 8).
- **Pixels:** 8 pixels per byte. **Bit 7** = first pixel of the 8, **bit 0** = last (MSB first within each byte).
- **Meaning:** 1 = valve on (drop), 0 = valve off.

Example: for 64×40, send 40×8 = **320 bytes**, row 0 bytes 0–7, row 1 bytes 8–15, etc.

---

## Summary

1. Open serial at **115200**; optionally wait for `"START"`.
2. Send **3 bytes:** `image_h`, `valve_on_time`, `drawing_depth`.
3. For each frame:
   - **Re-drop:** send `d`.
   - **New image:** send `s` then `image_h × 8` bytes (row-major, 8 px/byte, MSB first).
4. Optionally use `g` (every 8 bytes) for flow control.
5. Wait for **`r`** before the next command.

**Note:** `e` (END_KEY) is defined in the Arduino code but not used; it is not part of this API.
