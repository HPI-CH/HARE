from datetime import datetime

import wandb

now_str = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')


def init_wandb(experiment_name: str, name: str, reinit: bool = True, group: str = None, *,
               project: str = 'Bachelorarbeit'):
    return wandb.init(project=project,
                      reinit=reinit,
                      entity="franzsw",
                      name=f"{name}-{now_str}",
                      group=group or f"{experiment_name}-{now_str}",
                      job_type="eval",
                      settings=wandb.Settings(start_method='fork'),
                      )
