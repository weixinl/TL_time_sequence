import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# SIGNAL_TYPES = [
#         "body_acc_x_",
#         "body_acc_y_",
#         "body_acc_z_",
#         "body_gyro_x_",
#         "body_gyro_y_",
#         "body_gyro_z_",
#         "total_acc_x_",
#         "total_acc_y_",
#         "total_acc_z_"
#     ]

# UCIHAR_LABEL_NUM=6
# UCIHAR_SUBJECT_NUM=30
INF=1e8
DEVICE_NUM=8