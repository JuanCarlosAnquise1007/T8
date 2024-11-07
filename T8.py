import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pulp

# Configuración de colores personalizados
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD', '#D4A5A5']

# Configuración global de matplotlib
plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# ===========================
# Ejercicio 8.1: Dakin's Branch and Bound
# ===========================
st.header("Ejercicio 8.1: Dakin's Branch and Bound")
st.write("Maximizar P(x1, x2, x3) = 4x1 + 3x2 + 3x3 sujeto a las restricciones:")
st.latex(r"""
\begin{cases}
4x_1 + 2x_2 + x_3 \leq 10 \\ 
3x_1 + 4x_2 + 2x_3 \leq 14 \\ 
2x_1 + x_2 + 3x_3 \leq 7 \\
x_1, x_2, x_3 \geq 0 \quad \text{y enteros}
\end{cases}
""")

# Definir y resolver el problema
prob_8_1 = pulp.LpProblem("Maximization_8_1", pulp.LpMaximize)
x1 = pulp.LpVariable('x1', lowBound=0, cat='Integer')
x2 = pulp.LpVariable('x2', lowBound=0, cat='Integer')
x3 = pulp.LpVariable('x3', lowBound=0, cat='Integer')
prob_8_1 += 4 * x1 + 3 * x2 + 3 * x3
prob_8_1 += 4 * x1 + 2 * x2 + x3 <= 10
prob_8_1 += 3 * x1 + 4 * x2 + 2 * x3 <= 14
prob_8_1 += 2 * x1 + x2 + 3 * x3 <= 7
prob_8_1.solve()

# Mostrar solución
st.write("Estado:", pulp.LpStatus[prob_8_1.status])
st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
st.write("Valor máximo de P =", pulp.value(prob_8_1.objective))

# Función para crear gráficos personalizados
def create_custom_plot(ax, title=""):
    x_vals = np.linspace(0, 5, 200)
    ax.plot(x_vals, (10 - 4 * x_vals) / 2, color=custom_colors[0], 
            label="4x1 + 2x2 + x3 ≤ 10", linewidth=2)
    ax.plot(x_vals, (14 - 3 * x_vals) / 4, color=custom_colors[1], 
            label="3x1 + 4x2 + 2x3 ≤ 14", linewidth=2)
    ax.plot(x_vals, (7 - 2 * x_vals), color=custom_colors[2], 
            label="2x1 + x2 + 3x3 ≤ 7", linewidth=2)
    
    feasible_region = np.minimum(np.minimum((10 - 4 * x_vals) / 2, 
                                          (14 - 3 * x_vals) / 4), 
                                (7 - 2 * x_vals))
    ax.fill_between(x_vals, 0, feasible_region, color=custom_colors[3], alpha=0.3)
    
    ax.set_xlim([0, 5])
    ax.set_ylim([0, 5])
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title)
    ax.legend()

# Gráfico ejercicio 8.1
fig, ax = plt.subplots()
create_custom_plot(ax, "Región Factible - Ejercicio 8.1")
st.pyplot(fig)
plt.close()

# ===========================
# Ejercicio 8.2: Resolución como LP e IP
# ===========================
st.header("Ejercicio 8.2: Resolución como LP e IP")
st.write("Resolver el Ejercicio 8.1 relajando las variables (sin enteros) y luego resolviendo con enteros:")

# Resolver como LP sin enteros
prob_8_2_lp = prob_8_1.copy()
x1.cat, x2.cat, x3.cat = 'Continuous', 'Continuous', 'Continuous'
prob_8_2_lp.solve()
st.write("Solución LP (sin enteros):")
st.write(f"x1 = {x1.varValue}, x2 = {x2.varValue}, x3 = {x3.varValue}")
st.write("Valor máximo de P LP =", pulp.value(prob_8_2_lp.objective))

# Gráfico ejercicio 8.2
fig, ax = plt.subplots()
create_custom_plot(ax, "Región Factible LP - Ejercicio 8.2")
st.pyplot(fig)
plt.close()

# ===========================
# Ejercicio 8.3: Minimización con Planos de Corte
# ===========================
st.header("Ejercicio 8.3: Minimización con Planos de Corte")
st.write("Minimizar C(x, y) = x - y sujeto a las restricciones:")
st.latex(r"""
\begin{cases}
3x + 4y \leq 6 \\
x - y \leq 1 \\
x, y \geq 0 \quad \text{y enteros}
\end{cases}
""")

# Definir y resolver el problema
x = pulp.LpVariable('x', lowBound=0, cat='Integer')
y = pulp.LpVariable('y', lowBound=0, cat='Integer')
prob_8_3 = pulp.LpProblem("Minimization_8_3", pulp.LpMinimize)
prob_8_3 += x - y
prob_8_3 += 3 * x + 4 * y <= 6
prob_8_3 += x - y <= 1
prob_8_3.solve()

st.write("Estado:", pulp.LpStatus[prob_8_3.status])
st.write(f"x = {x.varValue}, y = {y.varValue}")
st.write("Valor mínimo de C =", pulp.value(prob_8_3.objective))

# Gráfico ejercicio 8.3
fig, ax = plt.subplots()
x_vals = np.linspace(0, 3, 100)
ax.plot(x_vals, (6 - 3 * x_vals) / 4, color=custom_colors[0], 
        label="3x + 4y ≤ 6", linewidth=2)
ax.plot(x_vals, x_vals - 1, color=custom_colors[1], 
        label="x - y ≤ 1", linewidth=2)
ax.fill_between(x_vals, 0, np.minimum((6 - 3 * x_vals) / 4, x_vals - 1), 
                color=custom_colors[3], alpha=0.3)
ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Región Factible - Ejercicio 8.3")
ax.legend()
st.pyplot(fig)
plt.close()

# ===========================
# Ejercicio 8.5: Selección de Proyectos
# ===========================
st.header("Ejercicio 8.5: Selección de Proyectos")
st.write("Seleccionar proyectos para maximizar el NPV bajo restricciones de presupuesto anual")

# Definir y resolver el problema binario
prob_8_5 = pulp.LpProblem("Maximization_8_5", pulp.LpMaximize)
p1 = pulp.LpVariable("p1", 0, 1, cat="Binary")
p2 = pulp.LpVariable("p2", 0, 1, cat="Binary")
p3 = pulp.LpVariable("p3", 0, 1, cat="Binary")
p4 = pulp.LpVariable("p4", 0, 1, cat="Binary")
p5 = pulp.LpVariable("p5", 0, 1, cat="Binary")
p6 = pulp.LpVariable("p6", 0, 1, cat="Binary")
prob_8_5 += 141 * p1 + 187 * p2 + 121 * p3 + 83 * p4 + 262 * p5 + 127 * p6
prob_8_5 += 75 * p1 + 90 * p2 + 60 * p3 + 30 * p4 + 100 * p5 + 50 * p6 <= 250
prob_8_5 += 25 * p1 + 35 * p2 + 15 * p3 + 20 * p4 + 25 * p5 + 20 * p6 <= 75
prob_8_5 += 20 * p1 + 0 * p2 + 15 * p3 + 10 * p4 + 20 * p5 + 10 * p6 <= 50
prob_8_5 += 15 * p1 + 0 * p2 + 15 * p3 + 5 * p4 + 20 * p5 + 30 * p6 <= 50
prob_8_5 += 10 * p1 + 30 * p2 + 15 * p3 + 5 * p4 + 20 * p5 + 40 * p6 <= 50
prob_8_5.solve()

st.write("Proyectos seleccionados:")
for var in [p1, p2, p3, p4, p5, p6]:
    st.write(f"{var.name} = {var.varValue}")
st.write("Valor máximo de NPV =", pulp.value(prob_8_5.objective))

# Gráfico de barras para proyectos
fig, ax = plt.subplots()
proyectos = ["P1", "P2", "P3", "P4", "P5", "P6"]
valores = [p1.varValue, p2.varValue, p3.varValue, p4.varValue, p5.varValue, p6.varValue]
bars = ax.bar(proyectos, valores, color=custom_colors)

# Añadir valores sobre las barras
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom')

ax.set_title("Selección de Proyectos - Ejercicio 8.5")
ax.set_ylabel("Selección (1 = Sí, 0 = No)")
ax.set_xlabel("Proyectos")
st.pyplot(fig)
plt.close()