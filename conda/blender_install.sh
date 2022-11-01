cd ..
wget https://download.blender.org/release/Blender2.83/blender-2.83.20-linux-x64.tar.xz
sudo tar -xvf blender-2.83.20* --strip-components=1 -C /bin
rm -rf blender-2.83.20*
rm -rf blender-2.83.20*
sudo rm -rf /bin/2.83/python
sudo ln -s /home/$USER/anaconda3/envs/ns_ap /bin/2.83/python
