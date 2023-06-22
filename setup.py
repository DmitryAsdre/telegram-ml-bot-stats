
from distutils.core import setup
setup(
  name = 'tg_bot_ml',
  packages = ['tg_bot_ml'],
  version = '0.0.5',
  license='GPLv3',
  description = 'Simple telegram bot for logging ML statistics.',
  author = 'Dmitry',
  author_email = 'michalych2014@yandex.ru',
  url = 'https://github.com/DmitryAsdre/telegram-ml-bot-stats.git', 
  download_url = 'https://github.com/DmitryAsdre/telegram-ml-bot-stats/archive/refs/heads/main.zip',
  keywords = ['Telegram', 'Bot', 'ML', 'Machine Learning', 'Logger'],
  install_requires=[
          'pandas',
          'numpy',
          'matplotlib',
          'pyTelegramBotAPI',
          'pyyaml',
          'opencv-python', 
          'tabulate'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',  
    'Programming Language :: Python :: 3',      
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8', 
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11'
  ],
)