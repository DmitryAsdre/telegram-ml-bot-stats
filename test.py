import telebot
import random
import pandas as pd
import numpy as np
from tg_bot_ml.utils import create_credentials
import cv2
from time import sleep
import matplotlib.pyplot as plt
from tg_bot_ml.table_bot import TGTableSummaryWriter
from tg_bot_ml.img_bot import TGImgSummaryWriter

writer = TGImgSummaryWriter('./credentials.yaml', experiment_name='exp_test')
writer.add_description("Testing table writer")

for epoch in range(10):

    for i in range(100):
        i = 100*epoch + i
        writer.add_scalar('MSE', i**2, i)
        writer.add_scalar('RMSE', np.sqrt(i), i)
        writer.add_scalar('AUC', random.randint(0, 10) / 10, i)
    
    sleep(180)

    writer.send(send_images=False, send_plots=True)