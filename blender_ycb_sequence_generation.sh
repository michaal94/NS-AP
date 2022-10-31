#!/bin/bash
if [ $1 == "blender" ]; then
    echo "Running demo in Blender"
    blender -b -noaudio --python demo_sequence_generation_blender_ycb.py -- --input_instruction_json ./output_ycb/NS_AP_ycb_instructions_$2.json --input_scene_dir ./output_ycb/scenes --output_dir ./output_ycb_blender --instruction_idx $3
elif [ $1 == "mujoco" ]; then
    echo "Running with MuJoCo display"
    python demo_sequence_generation_blender_ycb.py --input_instruction_json ./output_ycb/NS_AP_ycb_instructions_$2.json --input_scene_dir ./output_ycb/scenes --output_dir ./output_ycb_blender --instruction_idx $3
else
    echo "Provide an argument where to display output: [blender, mujoco]"
fi