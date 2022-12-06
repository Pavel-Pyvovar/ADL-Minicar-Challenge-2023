# ADL-Minicar-Challenge-2023
Code base for ADL Minicar Challenge 2023

## How to start autopilot
- `cd donkeycar`
- `python manage.py drive --js --model <model> --type <model type>` (e.g. `python ../mycar/manage.py drive --js --model ../mycar/models/pilot_22-11-03_2.tflite --type tflite_linear`)
- Set **Local Angle** Drive Mode (by using the `start` button on a joystick)
- Set the constant speed (by using the up and down arrows on a joystick)
- Press the `Back` button to start the autopilot