root=path_to_download_video_splits
target=path_to_save_sliced_videos
mkdir ${root}/${target}_segs
for file in $(ls ${root}/${target}/); do
  scenedetect --input ${root}/${target}/${file} --output ${root}/${target}_segs/ detect-content split-video
done
