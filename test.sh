#!/bin/bash
if [ $1 == "blender" ]; then
    echo "Running demo in Blender"
    blender -b -noaudio --python test_grasp.py -- --input_instruction_json ./output_shared/NS_AP_demo_instructions.json --input_scene_dir ./output_shared/scenes --output_dir ./output_shared
elif [ $1 == "mujoco" ]; then
    echo "Running with MuJoCo display"
    python test_grasp.py --input_instruction_json ./output_shared/NS_AP_demo_instructions.json --input_scene_dir ./output_shared/scenes --output_dir ./output_shared
else
    echo "Provide an argument where to display output: [blender, mujoco]"
fi