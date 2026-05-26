class SmoothValue:

    def __init__(self, alpha=0.4):

        self.alpha = alpha

        self.value = None

    def update(self, current_value):

        if self.value is None:
            self.value = current_value

        else:
            self.value = (
                self.alpha * current_value
                + (1 - self.alpha) * self.value
            )

        return int(self.value)