# vibecoding

Auto-commit setup for this repo. See `scripts/auto_commit_start.sh`.

## PC OpenMV + STM32 Laser Grid

See `pc_app/` for the Python application.

### Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pc_app/main.py
```

### Protocol alignment (STM32F103 SDK notes)

- PC sends yaw/pitch angles in degrees as CSV: `YAW,PITCH\n`.
- STM32 firmware should interpret these as **gimbal RPY yaw/pitch**, then map to servo raw angles.
- The SDK manual indicates a **linear mapping** between gimbal RPY and servo raw angles, calibrated via 2-point or multi-point sampling.
- If your STM32 firmware already uses `Gimbal_SetYaw/Gimbal_SetPitch`, keep it as-is and just parse the CSV.
