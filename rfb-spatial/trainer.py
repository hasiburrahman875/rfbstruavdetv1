import os
command = ("python train.py "
           "--data data/aot.yaml "
           "--device 0 "
          #  "--batch-size 8 "
          #  "--epochs 100 "
          #  "--img 1280 "
          #  "--hyp data/hyps/hyp.VisDrone.yaml "
          #  "--cfg /cluster/pixstor/madrias-lab/Hasibur/Models/rf-aod/models/yolov5l-xs-tph-rf.yaml "
          #  "--name rf-aod-aot "
          #  "--adam "
           "--weights /cluster/pixstor/madrias-lab/Hasibur/Models/rf-aod/runs/train/rf-aod-aot/weights/last.pt "
           "--noautoanchor "
           "--resume "
        # #    "--exist-ok "
        
          )

os.system(command)
