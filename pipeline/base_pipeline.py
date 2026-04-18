class BasePipeline:
    def load_input(self, file):
        raise NotImplementedError
    
    def preprocess(self, data):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
    
    def explain(self, data):
        raise NotImplementedError
    
    def run(self, file):
        # Main flow - yeh kabhi nahi badlega
        data = self.load_input(file)
        data = self.preprocess(data)
        result = self.predict(data)
        explanation = self.explain(data)
        return result, explanation