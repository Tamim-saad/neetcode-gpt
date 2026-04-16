class Solution:
    def f(self, x):
      return x**2
    
    def derivative(self, f, x, h=1e-6):
      return (f(x+h)-f(x-h))/(2*h)
      

    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        
        for i in range(iterations):
          init = init - learning_rate*self.derivative(self.f,init)
        
        return round(init,5)
        
        pass
