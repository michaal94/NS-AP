cd scene_generation
blender -b -noaudio --python render_images_ycb.py -- -c ../ycb_cfg/scene_generation.json
cd ..