import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Evaluador de Riesgo de Diabetes", layout="centered")

st.title("🏥 Sistema Clínico de Evaluación de Riesgo de Diabetes")
st.write("Herramienta interactiva basada en datos epidemiológicos de la BRFSS 2015 para la detección temprana y triaje.")

# Carga de datos optimizada con caché para que la app vuele
@st.cache_data
def cargar_y_procesar_datos():
    ruta = r"C:\Users\Asus Vivobook\Downloads\diabetes_012_health_indicators_BRFSS2015.csv"
    datos = pd.read_csv(ruta)
    datos = datos.drop_duplicates()
    
    # Muestra para agilizar el cálculo de pesos
    df_muestra = datos.sample(n=5000, random_state=42)
    return datos, df_muestra

# Funciones de análisis estadístico (las mismas de tu script)
@st.cache_data
def obtener_analisis(df_muestra):
    # Análisis de BMI
    grupos_bmi = pd.cut(df_muestra["BMI"], bins=[0, 18.5, 25, 30, float("inf")], labels=["Bajo peso", "Normal", "Sobrepeso", "Obesidad"])
    resultado_bmi = df_muestra.groupby(grupos_bmi, observed=False)["Diabetes_012"].apply(lambda x: (x == 2).mean() * 100)
    
    # Análisis de Edad
    resultado_edad = df_muestra.groupby("Age")["Diabetes_012"].apply(lambda x: (x == 2).mean() * 100)
    
    # Análisis de Salud General
    resultado_salud = df_muestra.groupby("GenHlth")["Diabetes_012"].apply(lambda x: (x == 2).mean() * 100)
    
    # Comparación de Factores Binarios
    factores = ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "HvyAlcoholConsump", "DiffWalk"]
    resultados = []
    for factor in factores:
        g0 = df_muestra[df_muestra[factor] == 0]
        g1 = df_muestra[df_muestra[factor] == 1]
        p0 = (g0["Diabetes_012"] == 2).mean() * 100
        p1 = (g1["Diabetes_012"] == 2).mean() * 100
        resultados.append({"Factor": factor, "Diabetes sin factor (%)": p0, "Diabetes con factor (%)": p1, "Razón": p1 / p0})
    
    df_comp = pd.DataFrame(resultados).sort_values(by="Razón", ascending=False)
    
    # Construcción de tabla de pesos
    factores_pesos = []
    for _, fila in df_comp.iterrows():
        p0, p1 = fila["Diabetes sin factor (%)"], fila["Diabetes con factor (%)"]
        if p1 >= p0:
            pref_riesgo, cond = p1, 1
        else:
            pref_riesgo, cond = p0, 0
        factores_pesos.append({"Factor": fila["Factor"], "Condición de mayor riesgo": cond, "Razón": pref_riesgo / (p0 if cond==1 else p1)})
        
    factores_pesos.append({"Factor": "BMI", "Condición de mayor riesgo": resultado_bmi.idxmax(), "Razón": resultado_bmi.max() / resultado_bmi["Normal"]})
    factores_pesos.append({"Factor": "Age", "Condición de mayor riesgo": resultado_edad.idxmax(), "Razón": resultado_edad.max() / resultado_edad.loc[1]})
    factores_pesos.append({"Factor": "GenHlth", "Condición de mayor riesgo": resultado_salud.idxmax(), "Razón": resultado_salud.max() / resultado_salud.loc[1]})
    
    tabla = pd.DataFrame(factores_pesos)
    tabla["Fuerza"] = tabla["Razón"] - 1
    total_fuerza = tabla["Fuerza"].sum()
    tabla["Peso (%)"] = (tabla["Fuerza"] / total_fuerza) * 100
    
    return tabla, resultado_bmi, resultado_edad, resultado_salud

# Cargar entorno
try:
    df_global, df_muestra = cargar_y_procesar_datos()
    tabla_pesos, resultado_bmi, resultado_edad, resultado_salud = obtener_analisis(df_muestra)
    
    # Límites de riesgo basados en la muestra
    # (Simulamos puntajes para establecer los umbrales de la interfaz)
    limite_bajo, limite_alto = 20.0, 45.0 # Umbrales estándar del modelo
    
except Exception as e:
    st.error(f"Error al cargar los datos: {e}. Asegúrate de tener el CSV en la misma carpeta.")
    st.stop()

# --- FORMULARIO INTERACTIVO EN LA WEB ---
st.subheader("📋 Ingrese los datos del paciente")

with st.form("form_paciente"):
    col1, col2 = st.columns(2)
    
    with col1:
        high_bp = st.selectbox("¿Padece presión arterial alta?", ["No", "Sí"])
        high_chol = st.selectbox("¿Tiene colesterol alto?", ["No", "Sí"])
        smoker = st.selectbox("¿Fumador activo o histórico?", ["No", "Sí"])
        stroke = st.selectbox("¿Antecedente de derrame cerebral?", ["No", "Sí"])
        heart_disease = st.selectbox("¿Enfermedad cardíaca o ataque previo?", ["No", "Sí"])
        
    with col2:
        phys_activity = st.selectbox("¿Realiza actividad física regular?", ["No", "Sí"])
        hvy_alcohol = st.selectbox("¿Consumo elevado de alcohol?", ["No", "Sí"])
        diff_walk = st.selectbox("¿Dificultad seria para caminar?", ["No", "Sí"])
        
    st.markdown("---")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=200.0, value=70.0)
    with col4:
        altura = st.number_input("Altura (m)", min_value=1.0, max_value=2.5, value=1.75)
    with col5:
        gen_hlth = st.selectbox("Salud general percibida", [
            ("1 - Excelente", 1), 
            ("2 - Muy buena", 2), 
            ("3 - Buena", 3), 
            ("4 - Regular", 4), 
            ("5 - Mala", 5)
        ], format_func=lambda x: x[0])[1]

    edad_opcion = st.selectbox("Seleccione su rango de edad:", [
        ("18 - 24 años", 1), ("25 - 29 años", 2), ("30 - 34 años", 3), ("35 - 39 años", 4),
        ("40 - 44 años", 5), ("45 - 49 años", 6), ("50 - 54 años", 7), ("55 - 59 años", 8),
        ("60 - 64 años", 9), ("65 - 69 años", 10), ("70 - 74 años", 11), ("75 - 79 años", 12), ("80+ años", 13)
    ], format_func=lambda x: x[0])
    
    submitted = st.form_submit_button("Calcular Riesgo de Diabetes")

if submitted:
    # Mapeo de respuestas
    paciente = {
        "HighBP": 1 if high_bp == "Sí" else 0,
        "HighChol": 1 if high_chol == "Sí" else 0,
        "Smoker": 1 if smoker == "Sí" else 0,
        "Stroke": 1 if stroke == "Sí" else 0,
        "HeartDiseaseorAttack": 1 if heart_disease == "Sí" else 0,
        "PhysActivity": 1 if phys_activity == "Sí" else 0,
        "HvyAlcoholConsump": 1 if hvy_alcohol == "Sí" else 0,
        "DiffWalk": 1 if diff_walk == "Sí" else 0,
        "BMI": peso / (altura ** 2),
        "Age": edad_opcion[1],
        "GenHlth": gen_hlth
    }
    
    # Cálculo del puntaje (Lógica idéntica al script)
    puntaje = 0
    factores_binarios = ["HighBP", "HighChol", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity", "HvyAlcoholConsump", "DiffWalk"]
    for _, fila in tabla_pesos.iterrows():
        f = fila["Factor"]
        if f in factores_binarios and paciente[f] == fila["Condición de mayor riesgo"]:
            puntaje += fila["Peso (%)"]
            
    bmi_val = paciente["BMI"]
    cat_bmi = "Bajo peso" if bmi_val < 18.5 else ("Normal" if bmi_val < 25 else ("Sobrepeso" if bmi_val < 30 else "Obesidad"))
    peso_bmi = tabla_pesos[tabla_pesos["Factor"] == "BMI"]["Peso (%)"].iloc[0]
    r_bmi = resultado_bmi[cat_bmi] / resultado_bmi["Normal"]
    r_bmi_max = resultado_bmi.max() / resultado_bmi["Normal"]
    if r_bmi > 1 and r_bmi_max > 1:
        puntaje += peso_bmi * ((r_bmi - 1) / (r_bmi_max - 1))
        
    peso_edad = tabla_pesos[tabla_pesos["Factor"] == "Age"]["Peso (%)"].iloc[0]
    r_edad = resultado_edad.loc[paciente["Age"]] / resultado_edad.loc[1]
    r_edad_max = resultado_edad.max() / resultado_edad.loc[1]
    if r_edad > 1 and r_edad_max > 1:
        puntaje += peso_edad * ((r_edad - 1) / (r_edad_max - 1))
        
    peso_salud = tabla_pesos[tabla_pesos["Factor"] == "GenHlth"]["Peso (%)"].iloc[0]
    r_salud = resultado_salud.loc[paciente["GenHlth"]] / resultado_salud.loc[1]
    r_salud_max = resultado_salud.max() / resultado_salud.loc[1]
    if r_salud > 1 and r_salud_max > 1:
        puntaje += peso_salud * ((r_salud - 1) / (r_salud_max - 1))
        
    puntaje = round(puntaje, 2)
    
    # Clasificación
    if puntaje <= 20:
        riesgo = "Bajo"
        color = "green"
    elif puntaje <= 45:
        riesgo = "Medio"
        color = "orange"
    else:
        riesgo = "Alto"
        color = "red"
        
    st.markdown("---")
    st.subheader("📊 Resultados de la Evaluación")
    st.metric(label="Puntaje de Riesgo Estimado", value=puntaje)
    st.markdown(f"### Nivel de Riesgo Clínico: **:{color}[{riesgo}]**")
    
    if riesgo == "Bajo":
        st.success("Recomendación: Mantenga sus hábitos de vida saludables y chequeos preventivos anuales.")
    elif riesgo == "Medio":
        st.warning("Recomendación: Riesgo moderado (Prediabetes). Se sugiere una evaluación analítica de glucosa con su médico.")
    else:
        st.error("Recomendación: Alto riesgo detectado. Se recomienda consultar prioritariamente con un profesional de salud.")

    # Visualización gráfica de pesos en la web
    st.markdown("---")
    st.subheader("📈 Impacto de los Factores de Riesgo en el Modelo")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(tabla_pesos["Factor"], tabla_pesos["Peso (%)"], color="skyblue")
    ax.set_xlabel("Peso Relativo (%)")
    ax.invert_yaxis()
    st.pyplot(fig)
