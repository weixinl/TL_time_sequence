
import utils
from my_env import *
import models
import tasks



if __name__ == "__main__":
    torch.manual_seed(10)
    tasks.ucihar_small_baseline_task()
    # tasks.ucihar_small_transfer_task()


