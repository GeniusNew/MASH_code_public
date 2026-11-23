
class ModelResponseError(ValueError):
    def __init__(self, message, *, content=None, position=None):
        super().__init__(message)
        self.content = content
        self.position = position
