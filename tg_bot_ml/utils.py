import os
import requests
import yaml


def get_chat_id(bot_token, chat_name):
    message = "trying to find out chat id"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage?chat_id=@{chat_name}&text={message}"

    res = requests.get(url).json()

    return res['result']['chat']['id']

def create_credentials(bot_token, chat_name, PATH_TO_SAVE=None):
    chat_id = get_chat_id(bot_token=bot_token, chat_name=chat_name)

    credentials = {"TOKEN" : bot_token,
                   "CHAT_ID" : chat_id,
                   "CHAT_NAME" : chat_name}
    
    if PATH_TO_SAVE:
        with open(PATH_TO_SAVE, 'w') as write_credentials:
            yaml.dump(credentials, write_credentials)

    return credentials