#!/bin/bash
if [ $1 == "blender" ]; then
    echo "Running demo in Blender"
    blender -b -noaudio --python test_stiffness.py -- --input_instruction_json ./output_ycb/NS_AP_ycb_instructions_2_stiffness.json --input_scene_dir ./output_ycb/scenes --output_dir ./output_asd
elif [ $1 == "mujoco" ]; then
    echo "Running with MuJoCo display"
    python test_stiffness.py --input_instruction_json ./output_ycb/NS_AP_ycb_instructions_2_stiffness.json --input_scene_dir ./output_ycb/scenes --output_dir ./output_asd
else
    echo "Provide an argument where to display output: [blender, mujoco]"
fi