

class DetectorInterface:
    def train(self, trainData):
      """Train using many sessions"""
      pass

    def trainSession(self, session):
        """Train parameters of the model"""
        pass

    def evaluate(self, session):
        """Evaluate session"""
        pass
