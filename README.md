# RaspiCode — Victim Detection for RoboCupJunior Rescue Maze

Vision system running on a Raspberry Pi for a robot competing in **RoboCupJunior Rescue Maze**.

## What is Rescue Maze?

RoboCupJunior Rescue Maze is an autonomous robotics competition where a robot must navigate a multi-level maze filled with obstacles, ramps, and dead ends — without any human control. The maze contains victims that the robot must find and correctly identify to score points ([Official 2026 Rules](https://junior.robocup.org/wp-content/uploads/2026/02/RCJRescueMaze2026-final.pdf)).

## Victim Types (2026 Rules)

There are two types of victims placed on the maze walls, about 7 cm above the floor:

### 1. Letter Victims (Greek letters)

Black uppercase Greek letters printed in a sans-serif font, 4 cm tall. They can be rotated.

| Letter | Status   | Rescue kits | Points (linear tile) | Points (floating tile) |
|--------|----------|-------------|----------------------|------------------------|
| **Φ**  | Harmed   | up to 2     | 5 pts + up to 30     | 15 pts + up to 30      |
| **Ψ**  | Stable   | up to 1     | 5 pts + up to 10     | 15 pts + up to 10      |
| **Ω**  | Unharmed | 0           | 5 pts                | 15 pts                 |

### 2. Cognitive Targets (colored concentric rings)

A bullseye-shaped circle with 5 concentric rings, outermost diameter 5 cm. Each ring has a color that maps to a value:

| Ring Color | Value |
|------------|-------|
| Blue       | +2    |
| Green      | +1    |
| Yellow     |  0    |
| Red        | −1    |
| Black      | −2    |

The health status is determined by summing all 5 ring values:

| Sum | Status   | Rescue kits | Points (linear tile) | Points (floating tile) |
|-----|----------|-------------|----------------------|------------------------|
| 2   | Harmed   | up to 2     | 10 pts + up to 30    | 30 pts + up to 30      |
| 1   | Stable   | up to 1     | 10 pts + up to 10    | 30 pts + up to 10      |
| 0   | Unharmed | 0           | 10 pts               | 30 pts                 |

To score, the robot must stop within 15 cm of a victim and blink its indicator LED for 5 seconds (500 ms on / 500 ms off).

> **Note:** The code internally uses the labels `H`, `S`, `U` — carry-overs from the pre-2026 rules where victims were the letters H, S, U. The 2026 rules replaced these with Φ, Ψ, Ω and introduced cognitive targets instead of plain colored markers.

## Detection Approach

This code runs two camera threads in parallel (left and right camera) and uses two complementary detection methods per frame:

**1. TFLite Letter Classification**
A quantized `.tflite` model classifies grayscale camera frames to detect the Greek letter victims on the walls.

**2. OpenCV Cognitive Target Analysis**
For circular cognitive targets, the code detects the contour of the bullseye, applies a perspective warp to get a flat top-down view, and scans five radii at 12 angles each. The dominant color per ring is mapped to its point value and summed to determine health status.

A result is only transmitted after 5 consistent detections within a 3-second window to avoid false positives.

## ToF Proximity Sensors

Two **VL53L0X** time-of-flight sensors are mounted next to the cameras to detect nearby walls or obstacles. They run in a background thread via `tof.py` and are non-blocking — both sensors are polled simultaneously so neither waits for the other.

### Usage

```python
import tof

tof.start()               # initializes sensors and starts background thread

tof.state["L"]            # 1 = object closer than threshold, 0 = clear
tof.state["R"]

tof.raw["L"]              # last measured distance in mm
tof.raw["R"]

tof.set_threshold(150)    # change threshold at runtime (resets to THRESHOLD_MM on restart)
```

### Pin Assignment

| Signal       | GPIO |
|--------------|------|
| XSHUT left   | 24   |
| XSHUT right  | 25   |
| IRQ left     | 5    |
| IRQ right    | 6    |
| SDA          | 2    |
| SCL          | 3    |

The sensors start at the default I2C address `0x29` and are reassigned to `0x2A` (left) and `0x2B` (right) on startup. The script handles restarts gracefully — if the sensors already hold their assigned addresses (XSHUT did not reset them), they are reused directly without re-initialization.

### Resource Usage

The ToF thread uses ~0% CPU and ~22 MB RAM. It has no measurable impact on camera detection throughput.

## Files

| File                   | Purpose                                              |
|------------------------|------------------------------------------------------|
| `main.py`              | Main detection loop — dual-camera threads, serial I/O|
| `tof.py`               | VL53L0X proximity sensor module (background thread)  |
| `tof_example.py`       | Minimal usage example for `tof.py`                   |
| `tof_diag.py`          | I2C diagnostic script                                |
| `datensatzGenerator.py`| Captures training images from a live camera feed     |
| `trained.tflite`       | Quantized TFLite model for letter detection          |
| `labels.txt`           | Class labels: background, H, S, U                   |

## Hardware

- Raspberry Pi (with GPIO)
- 2× Raspberry Pi Camera (Picamera2 API)
- 2× VL53L0X ToF distance sensor
- Serial connection to main robot controller (Arduino)

## Setup

```bash
# System package (not installable via pip)
sudo apt install python3-picamera2

# Python dependencies
pip install -r requirements.txt
```

Run the detection:
```bash
python3 mainWM.py
```

The script listens for serial commands from the Arduino:

| Command | Action                        |
|---------|-------------------------------|
| `<I>`   | Initialize / handshake        |
| `<E>`   | Enable both cameras           |
| `<D>`   | Disable both cameras          |
| `<RE>`  | Re-enable right camera only   |
| `<RD>`  | Disable right camera only     |

When a victim is identified, the Pi sends back e.g. `<LH>` (left camera, Harmed) or `<RS>` (right camera, Stable).
