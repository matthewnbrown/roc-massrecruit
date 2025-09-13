from numpy import random

# onmousemove = function(e)
# {console.log("mouse location:", e.clientX, e.clientY)}


class ROCCaptchaSelector():
    __btn_dimensions = (40, 30)
    __keypadTopLeft = {'roc_recruit': [890, 705],
                       'roc_armory': [973, 1011],
                       'roc_attack': [585, 680],
                       'roc_spy': [585, 695],
                       'roc_training': [973, 453]}
    __keypadGap = [52, 42]

    def __init__(self, resolution=None) -> None:
        self.resolution = resolution

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
