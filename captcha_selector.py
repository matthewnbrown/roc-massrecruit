from numpy import random
from settings_loader import get_settings

class ROCCaptchaSelector():
    def __init__(self, resolution=None, settings_file="settings.json") -> None:
        self.resolution = resolution
        self.settings = get_settings(settings_file)
        
        # Load configuration from settings
        captcha_config = self.settings.get_captcha_selector_config()
        self.__btn_dimensions = tuple(captcha_config['button_dimensions'])
        self.__keypadTopLeft = captcha_config['keypad_positions']
        self.__keypadGap = captcha_config['keypad_gap']

    def get_xy(self, number):
        pass

    def get_xy_static(self, number, page):
        if page not in self.__keypadTopLeft:
            raise Exception(
                f'Page {page} does not have coordinates for captchas!'
                )
        number = int(number) - 1
        x_btn = self.__keypadTopLeft[page][0] \
            + (number % 3) * self.__keypadGap[0]
        y_btn = self.__keypadTopLeft[page][1] \
            + (number // 3) * self.__keypadGap[1]

        x_click = -x_btn
        while x_click < x_btn or x_click > x_btn + self.__btn_dimensions[0]:
            x_click = x_btn + random.normal(0, self.__btn_dimensions[0]/3)
        y_click = -y_btn
        while y_click < y_btn or y_click > y_btn + self.__btn_dimensions[1]:
            y_click = y_btn + random.normal(0, self.__btn_dimensions[1]/3)

        return (int(x_click), int(y_click))
