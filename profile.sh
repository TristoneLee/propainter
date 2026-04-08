 PROPAINTER_PROFILE=1 python inference_propainter.py \
  --video inputs/video_completion/running_car.mp4 \
  --mask inputs/video_completion/mask_square.png \
  --height  720 \
  --width 1280 \
  --frames 80 \
  --fp16