rsync -azP ../NS-AP michal@s3:/home/michal/codes --exclude '.git' --include 'output/checkpoints/' --exclude 'output/*' --exclude 'test' 
