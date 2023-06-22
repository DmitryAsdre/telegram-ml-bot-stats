import yaml
import cv2

import telebot
from telebot.types import InputMediaPhoto



class TGSummaryWriterBase():
    def __init__(self, path_to_credentials, experiment_name):
        self.credentials = self.load_credentials(path_to_credentials)
        self.experiment_name = experiment_name
        self.bot = telebot.TeleBot(self.credentials['TOKEN'],
                                   parse_mode=None)
        self.topic = self.bot.create_forum_topic(self.credentials['CHAT_ID'], 
                                                 self.experiment_name)
        self.delimiter = "-"*90
        self.description = None

    def send_message(self, message):
        self.bot.send_message(self.credentials['CHAT_ID'], text=message, 
                              reply_to_message_id=self.topic.message_thread_id)
        
    def send_pandas_df(self, df):
        message = f"<pre>\n{df.to_markdown()}\n</pre>"
        self.bot.send_message(self.credentials["CHAT_ID"], text=message,
                              reply_to_message_id=self.topic.message_thread_id, parse_mode='HTML')
        
    def send_numpy_image(self, np_image):
        success, img_enc = cv2.imencode('.jpg', np_image)
        assert(success)
        
        img_enc = img_enc.tostring()

        self.bot.send_photo(self.credentials['CHAT_ID'], img_enc,
                            reply_to_message_id=self.topic.message_thread_id)
    
    def send_numpy_images(self, np_images):
        encoded_imgs = []
        for img in np_images:
            success, img_enc = cv2.imencode('.jpg', img)
            assert(success)
            
            img_enc = img_enc.tostring()
            encoded_imgs.append(InputMediaPhoto(img_enc))
        
        self.bot.send_media_group(self.credentials['CHAT_ID'], encoded_imgs, 
                                  reply_to_message_id=self.topic.message_thread_id)
    

    def send_delimiter(self):
        if self.description is not None:
            self.send_message(f"{self.delimiter} \n {self.description}")
        else:
            self.send_message(self.delimiter)

    @staticmethod
    def load_credentials(path_to_cred):
        with open(path_to_cred, 'r') as read_c:
            credentials = yaml.safe_load(read_c)
        return credentials

