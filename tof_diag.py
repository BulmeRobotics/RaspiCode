from smbus2 import SMBus, i2c_msg

bus = SMBus(1)
addr = 0x29

print(f"=== VL6180X Diagnose @ 0x{addr:02X} ===\n")

# Method A: combined write+read (repeated start)
try:
    w = i2c_msg.write(addr, [0x00, 0x00])
    r = i2c_msg.read(addr, 8)
    bus.i2c_rdwr(w, r)
    print(f"[A] Combined (repeated start)  0x000-0x007: {[hex(x) for x in list(r)]}")
except OSError as e:
    print(f"[A] Fehler: {e}")

# Method B: separate write, then separate read (stop-start)
try:
    w = i2c_msg.write(addr, [0x00, 0x00])
    bus.i2c_rdwr(w)
    r = i2c_msg.read(addr, 8)
    bus.i2c_rdwr(r)
    print(f"[B] Separate (stop-start)      0x000-0x007: {[hex(x) for x in list(r)]}")
except OSError as e:
    print(f"[B] Fehler: {e}")

# Method C: write_i2c_block_data + read_byte
try:
    bus.write_i2c_block_data(addr, 0x00, [0x00])
    val = bus.read_byte(addr)
    print(f"[C] block_data + read_byte     0x000      : {hex(val)}")
except OSError as e:
    print(f"[C] Fehler: {e}")

# Method D: read FRESH_OUT_OF_RESET (0x016) — should be 0x01 if just booted
try:
    w = i2c_msg.write(addr, [0x00, 0x16])
    r = i2c_msg.read(addr, 1)
    bus.i2c_rdwr(w, r)
    val = list(r)[0]
    print(f"[D] FRESH_OUT_OF_RESET (0x016): {hex(val)}  (erwartet 0x01)")
except OSError as e:
    print(f"[D] Fehler: {e}")

print("\nErwartet: Model-ID = 0xb4, FRESH_OUT_OF_RESET = 0x01")
bus.close()
