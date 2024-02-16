

class Relation:
    def __init__(self, from_text, to_text, direction, from_id=0, to_id=0, n_blanks=0):
        self.from_text = from_text
        self.to_text = to_text
        self.direction = direction
        self.from_id = from_id
        self.to_id = to_id
        self.n_blanks = n_blanks

    def equal(self, other, compare_blanks=True):
        return self.from_text == other.from_text \
               and self.to_text == other.to_text \
               and self.direction == other.direction \
               and (self.n_blanks == other.n_blanks if compare_blanks else True)

    def __repr__(self):
        direction = 'RIGHT' if self.direction == 1 else 'DOWN'
        if self.from_id == 0 and self.to_id == 0:
            return f"Relation({self.from_text}, {self.to_text}, {direction}, n_blanks={self.n_blanks})"
        else:
            return f"Relation({self.from_text}, {self.to_text}, {direction}, {self.from_id}, {self.to_id}, n_blanks={self.n_blanks})"
