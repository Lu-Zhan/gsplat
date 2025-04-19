import tyro
import time
import torch

from options import Config
from system import Runner


def main(cfg: Config):
    runner = Runner(cfg)
    # luzhan: convert render_with_bg to a boolean
    cfg.render_with_bg = cfg.render_with_bg == 1
    print(f"render_with_bg: {cfg.render_with_bg}")

    if cfg.ckpt is not None:
        # run eval only
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for k in runner.splats.keys():
            runner.splats[k].data = ckpt["splats"][k]

        # load hdr scaler and splats_bg
        try:
            runner.update_hdr_scaler(torch.exp(ckpt["hdr_scaler"]))
        except:
            print("hdr_scaler not found in checkpoint.")
        
        try:
            for k in runner.splats_bg.keys():
                runner.splats_bg[k].data = ckpt["splats_bg"][k]
        except:
            print("splats_bg not found in checkpoint.")
            
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)
