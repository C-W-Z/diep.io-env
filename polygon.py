from env import Unit, UnitType

class Polygon(Unit):
    def __init__(self, side = 4):
        match side:
            case 4: # square
                hp = 10
                body_damage = 8.0
                score = 10
            case 3: # triangle
                hp = 30
                body_damage = 8.0
                score = 25
            case 5: # pentagon
                hp = 100
                body_damage = 12.0
                score = 30
            case _:
                assert False, "Invalid polygon side count"

        super().__init__(
            type=UnitType.Polygon,
            max_hp = hp,
            body_damage=body_damage,
            score=score,
        )
