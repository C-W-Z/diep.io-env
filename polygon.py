from env import Unit

class Polygon(Unit):
    def __init__(self, side = 4):
        match side:
            case 4: # square
                hp = 10
            case 3: # triangle
                hp = 30
            case 5: # pentagon
                hp = 100
            case _:
                assert False, "Invalid polygon side count"

        super().__init__(max_hp = hp)