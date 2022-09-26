class ShopVrbXMLFiles:
    
    @staticmethod
    def get(name, mode='train'):
        if mode == 'train':
            return getattr(ShopVrbXMLFilesTrain, name)
        else:
            raise NotImplementedError

class ShopVrbXMLFilesTrain:
    baking_tray1 = './assets/train/baking_tray1.xml'
    baking_tray2 = './assets/train/baking_tray2.xml'
    blender1 = './assets/train/blender1.xml'
    blender2 = './assets/train/blender2.xml'
    bowl1 = './assets/train/bowl1.xml'
    bowl2 = './assets/train/bowl2.xml'
    bowl3 = './assets/train/bowl3.xml'
    chopping_board1 = './assets/train/chopping_board1.xml'
    chopping_board2 = './assets/train/chopping_board2.xml'
    chopping_board3 = './assets/train/chopping_board3.xml'
    coffee_maker1 = './assets/train/coffee_maker1.xml'
    coffee_maker2 = './assets/train/coffee_maker2.xml'
    food_box1 = './assets/train/food_box1.xml'
    food_box2 = './assets/train/food_box2.xml'
    fork1 = './assets/train/fork1.xml'
    fork2 = './assets/train/fork2.xml'
    glass1 = './assets/train/glass1.xml'
    glass2 = './assets/train/glass2.xml'
    glass3 = './assets/train/glass3.xml'
    glass4 = './assets/train/glass4.xml'
    glass5 = './assets/train/glass5.xml'
    kettle1 = './assets/train/kettle1.xml'
    kettle2 = './assets/train/kettle2.xml'
    knife1 = './assets/train/knife1.xml'
    knife2 = './assets/train/knife2.xml'
    mug1 = './assets/train/mug1.xml'
    mug2 = './assets/train/mug2.xml'
    mug3 = './assets/train/mug3.xml'
    pan1 = './assets/train/pan1.xml'
    pan2 = './assets/train/pan2.xml'
    plate1 = './assets/train/plate1.xml'
    plate2 = './assets/train/plate2.xml'
    pot1 = './assets/train/pot1.xml'
    pot2 = './assets/train/pot2.xml'
    scissors1 = './assets/train/scissors1.xml'
    scissors2 = './assets/train/scissors2.xml'
    soda_can = './assets/train/soda_can.xml'
    spoon1 = './assets/train/spoon1.xml'
    spoon2 = './assets/train/spoon2.xml'
    thermos1 = './assets/train/thermos1.xml'
    thermos2 = './assets/train/thermos2.xml'
    toaster1 = './assets/train/toaster1.xml'
    toaster2 = './assets/train/toaster2.xml'
    wine_glass1 = './assets/train/wine_glass1.xml'
    wine_glass2 = './assets/train/wine_glass2.xml'
    wine_glass3 = './assets/train/wine_glass3.xml'


class YCBXMLFiles:
    bleach_cleanser = './assets/ycb/bleach_cleanser.xml'
    cracker_box = './assets/ycb/cracker_box.xml'
    sugar_box = './assets/ycb/sugar_box.xml'
    mustard_bottle = './assets/ycb/mustard_bottle.xml'

    @staticmethod
    def get(name, mode='train'):
        return getattr(YCBXMLFiles, name)