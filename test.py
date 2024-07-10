from sympy import symbols, Function, sin, cos, sqrt, integrate

# 定义符号
phi_0, omega, t, dt, theta = symbols('phi_0 omega t dt theta')

# 定义函数 ω(τ)
omega_func = Function('omega')(t)

# 定义全微分表达式
total_differential = phi_0 * (-sin(theta) + sin(theta + integrate(omega_func, (t, t + dt))))

# 确定梯度分量
partial_derivative_omega = diff(total_differential, omega)
partial_derivative_t = diff(total_differential, t)

# 计算梯度的模
gradient_magnitude = sqrt(partial_derivative_omega**2 + partial_derivative_t**2)
gradient_magnitude.simplify()
