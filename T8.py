import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from autograd import grad
import autograd.numpy as anp

class GradientDescent:
    def __init__(self, learning_rate=0.01, momentum=0.9, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def optimize(self, func, gradient_func, initial_point):
        current_point = np.array(initial_point, dtype=float)
        velocity = np.zeros_like(current_point)
        
        history_values = [func(current_point)]
        history_points = [current_point.copy()]
        
        for i in range(self.max_iterations):
            gradient = gradient_func(current_point)
            velocity = self.momentum * velocity - self.learning_rate * gradient
            new_point = current_point + velocity
            
            history_points.append(new_point.copy())
            history_values.append(func(new_point))
            
            if np.linalg.norm(new_point - current_point) < self.tolerance:
                break
                
            current_point = new_point
            
        return current_point, history_values, history_points

# Función cuadrática generalizada
def quadratic_function(x, a, b):
    return a * x[0]**2 + b * x[1]**2

# Gradiente de la función cuadrática
def quadratic_gradient(x, a, b):
    return np.array([2 * a * x[0], 2 * b * x[1]])

def create_contour_data(x_range, y_range, func, a, b):
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]), a, b)
    
    return X, Y, Z

st.title("Visualización de Descenso de Gradiente")
st.write("""
Esta aplicación demuestra el algoritmo de descenso de gradiente con momento en una función cuadrática generalizada.
Ajusta los parámetros de la función y el algoritmo para ver cómo afectan a la convergencia.
""")

# Sidebar para parámetros de la función
st.sidebar.header("Parámetros de la Función Cuadrática")
a = st.sidebar.number_input("Coeficiente a (para x²)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
b = st.sidebar.number_input("Coeficiente b (para y²)", value=2.0, min_value=0.1, max_value=10.0, step=0.1)

# Sidebar para parámetros del algoritmo
st.sidebar.header("Parámetros del Algoritmo de Optimización")
learning_rate = st.sidebar.slider("Tasa de Aprendizaje", 0.01, 1.0, 0.1, 0.01)
momentum = st.sidebar.slider("Momento", 0.0, 0.99, 0.9, 0.01)
max_iterations = st.sidebar.slider("Máximo de Iteraciones", 100, 2000, 1000, 100)
initial_x = st.sidebar.number_input("Punto Inicial X", value=2.0, min_value=-5.0, max_value=5.0, step=0.1)
initial_y = st.sidebar.number_input("Punto Inicial Y", value=1.0, min_value=-5.0, max_value=5.0, step=0.1)

# Crear el optimizador con los parámetros seleccionados
optimizer = GradientDescent(
    learning_rate=learning_rate,
    momentum=momentum,
    max_iterations=max_iterations
)

# Optimizar
initial_point = np.array([initial_x, initial_y])
optimal_point, values_history, points_history = optimizer.optimize(
    lambda x: quadratic_function(x, a, b),
    lambda x: quadratic_gradient(x, a, b),
    initial_point
)

# Crear las visualizaciones
points_history = np.array(points_history)

# Crear dos columnas para las gráficas
col1, col2 = st.columns(2)

with col1:
    st.subheader("Convergencia de la Función Objetivo")
    fig1, ax1 = plt.subplots()
    ax1.plot(values_history)
    ax1.set_xlabel('Iteraciones')
    ax1.set_ylabel('Valor de la Función')
    ax1.grid(True)
    st.pyplot(fig1)

with col2:
    st.subheader("Trayectoria de Optimización")
    fig2, ax2 = plt.subplots()
    
    # Crear datos para el contour plot
    X, Y, Z = create_contour_data((-3, 3), (-3, 3), quadratic_function, a, b)
    ax2.contour(X, Y, Z, levels=20)
    ax2.plot(points_history[:, 0], points_history[:, 1], 'b.-')
    ax2.plot(points_history[0, 0], points_history[0, 1], 'go', label='Inicio')
    ax2.plot(points_history[-1, 0], points_history[-1, 1], 'ro', label='Final')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Mostrar resultados
st.subheader("Resultados")
col3, col4 = st.columns(2)
with col3:
    st.write(f"Punto óptimo encontrado: [{optimal_point[0]:.4f}, {optimal_point[1]:.4f}]")
with col4:
    st.write(f"Valor mínimo de la función: {quadratic_function(optimal_point, a, b):.4f}")

# Información adicional
st.write("""
### Explicación de los Parámetros:
- **Tasa de Aprendizaje**: Controla el tamaño de los pasos en cada iteración.
- **Momento**: Determina cuánto influye la dirección anterior en el siguiente paso.
- **Máximo de Iteraciones**: Límite de iteraciones para el algoritmo.
- **Punto Inicial**: Coordenadas (x,y) desde donde comienza la optimización.
""")

# Métricas adicionales
st.subheader("Métricas de Convergencia")
num_iterations = len(values_history) - 1
improvement = values_history[0] - values_history[-1]

col5, col6 = st.columns(2)
with col5:
    st.metric("Número de Iteraciones", num_iterations)
with col6:
    st.metric("Mejora Total", f"{improvement:.4f}")
