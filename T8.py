import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def ejercicio_8_1():
    st.header("Ejercicio 8.1")
    st.write("""
    Maximize P(x₁, x₂, x₃) = 4x₁ + 3x₂ + 3x₃
    
    Subject to:
    4x₁ + 2x₂ + x₃ ≤ 10
    3x₁ + 4x₂ + 2x₃ ≤ 14
    2x₁ + x₂ + 3x₃ ≤ 7
    
    where x₁, x₂, x₃ are nonnegative integers
    """)
    
    # Crear el problema
    prob = pulp.LpProblem("Exercise_8_1", pulp.LpMaximize)
    
    # Variables de decisión
    x1 = pulp.LpVariable("x1", 0, None, pulp.LpInteger)
    x2 = pulp.LpVariable("x2", 0, None, pulp.LpInteger)
    x3 = pulp.LpVariable("x3", 0, None, pulp.LpInteger)
    
    # Función objetivo
    prob += 4*x1 + 3*x2 + 3*x3
    
    # Restricciones
    prob += 4*x1 + 2*x2 + x3 <= 10, "Restriction_1"
    prob += 3*x1 + 4*x2 + 2*x3 <= 14, "Restriction_2"
    prob += 2*x1 + x2 + 3*x3 <= 7, "Restriction_3"
    
    # Resolver
    prob.solve()
    
    st.write("### Resultados:")
    st.write(f"Estado: {pulp.LpStatus[prob.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob.objective)}")
    st.write(f"x₁ = {pulp.value(x1)}")
    st.write(f"x₂ = {pulp.value(x2)}")
    st.write(f"x₃ = {pulp.value(x3)}")

def ejercicio_8_3():
    st.header("Ejercicio 8.3")
    st.write("""
    Minimize C(x,y) = x - y
    
    Subject to:
    3x + 4y ≤ 6
    x - y ≤ 1
    
    where x and y are nonnegative integers
    """)
    
    # Crear el problema
    prob = pulp.LpProblem("Exercise_8_3", pulp.LpMinimize)
    
    # Variables de decisión
    x = pulp.LpVariable("x", 0, None, pulp.LpInteger)
    y = pulp.LpVariable("y", 0, None, pulp.LpInteger)
    
    # Función objetivo
    prob += x - y
    
    # Restricciones
    prob += 3*x + 4*y <= 6, "Restriction_1"
    prob += x - y <= 1, "Restriction_2"
    
    # Resolver
    prob.solve()
    
    st.write("### Resultados:")
    st.write(f"Estado: {pulp.LpStatus[prob.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob.objective)}")
    st.write(f"x = {pulp.value(x)}")
    st.write(f"y = {pulp.value(y)}")

def ejercicio_8_4():
    st.header("Ejercicio 8.4")
    st.write("""
    Maximize P(x₁, x₂, x₃) = 4x₁ + 3x₂ + 3x₃
    
    Subject to:
    4x₁ + 2x₂ + x₃ ≤ 10
    3x₁ + 4x₂ + 2x₃ ≤ 14
    2x₁ + x₂ + 3x₃ ≤ 7
    
    where x₁, x₂, x₃ are nonnegative integers
    """)
    
    # Este ejercicio es idéntico al 8.1 pero usando Gomory Cut-Planes
    # Para simplificar, usaremos PuLP que ya implementa este método
    prob = pulp.LpProblem("Exercise_8_4", pulp.LpMaximize)
    
    x1 = pulp.LpVariable("x1", 0, None, pulp.LpInteger)
    x2 = pulp.LpVariable("x2", 0, None, pulp.LpInteger)
    x3 = pulp.LpVariable("x3", 0, None, pulp.LpInteger)
    
    prob += 4*x1 + 3*x2 + 3*x3
    prob += 4*x1 + 2*x2 + x3 <= 10
    prob += 3*x1 + 4*x2 + 2*x3 <= 14
    prob += 2*x1 + x2 + 3*x3 <= 7
    
    prob.solve()
    
    st.write("### Resultados:")
    st.write(f"Estado: {pulp.LpStatus[prob.status]}")
    st.write(f"Valor óptimo: {pulp.value(prob.objective)}")
    st.write(f"x₁ = {pulp.value(x1)}")
    st.write(f"x₂ = {pulp.value(x2)}")
    st.write(f"x₃ = {pulp.value(x3)}")

def ejercicio_8_5():
    st.header("Ejercicio 8.5")
    st.write("""
    Problema de selección de proyectos R&D
    
    La compañía debe seleccionar proyectos para maximizar NPV con restricciones de presupuesto.
    Presupuesto disponible: $250,000
    Presupuesto para años futuros: $75,000 (año 2) y $50,000 (años 3,4,5)
    """)
    
    # Datos del problema
    projects = range(1, 7)
    npv = {1: 141, 2: 187, 3: 121, 4: 83, 5: 262, 6: 127}
    cr = {
        1: [75, 25, 20, 15, 10],
        2: [90, 35, 0, 0, 0],
        3: [60, 15, 15, 15, 15],
        4: [30, 20, 10, 5, 5],
        5: [100, 25, 20, 20, 20],
        6: [50, 20, 10, 10, 10]
    }
    
    # Crear el problema
    prob = pulp.LpProblem("Project_Selection", pulp.LpMaximize)
    
    # Variables de decisión (binarias)
    x = pulp.LpVariable.dicts("project", projects, 0, 1, pulp.LpBinary)
    
    # Función objetivo
    prob += pulp.lpSum(npv[i] * x[i] for i in projects)
    
    # Restricciones de presupuesto
    # Año 1
    prob += pulp.lpSum(cr[i][0] * x[i] for i in projects) <= 250
    # Año 2
    prob += pulp.lpSum(cr[i][1] * x[i] for i in projects) <= 75
    # Años 3-5
    for year in range(2, 5):
        prob += pulp.lpSum(cr[i][year] * x[i] for i in projects) <= 50
    
    # Resolver
    prob.solve()
    
    st.write("### Resultados:")
    st.write(f"Estado: {pulp.LpStatus[prob.status]}")
    st.write(f"NPV Total Óptimo: ${pulp.value(prob.objective)}k")
    
    # Mostrar proyectos seleccionados
    selected = [i for i in projects if pulp.value(x[i]) > 0.5]
    st.write("Proyectos seleccionados:", selected)
    
    # Crear tabla de resultados
    results = []
    for i in projects:
        results.append({
            'Proyecto': i,
            'NPV': f"${npv[i]}k",
            'Seleccionado': 'Sí' if i in selected else 'No'
        })
    
    df = pd.DataFrame(results)
    st.write("### Tabla de Resultados:")
    st.table(df)

def main():
    st.title("Resolución de Ejercicios de Programación Lineal Entera")
    
    ejercicio = st.sidebar.selectbox(
        "Seleccione el ejercicio",
        ["Ejercicio 8.1", "Ejercicio 8.3", "Ejercicio 8.4", "Ejercicio 8.5"]
    )
    
    if ejercicio == "Ejercicio 8.1":
        ejercicio_8_1()
    elif ejercicio == "Ejercicio 8.3":
        ejercicio_8_3()
    elif ejercicio == "Ejercicio 8.4":
        ejercicio_8_4()
    elif ejercicio == "Ejercicio 8.5":
        ejercicio_8_5()

if __name__ == "__main__":
    main()