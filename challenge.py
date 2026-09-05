import pandas as pd
import matplotlib.pyplot as plt



# Carga el archivo CSV
def cargar_datos(ruta):
    datos = pd.read_csv(ruta)
    return datos


# Detecta valores nulos
def validar_nulos(df):
    nulos_por_columna = df.isnull().sum()
    total_nulos = nulos_por_columna.sum()
    return df


# Detecta y elimina registros duplicados
def validar_duplicados(df):
    cant_duplicados = df.duplicated().sum()

    # Opcional: Eliminar los duplicados
    df = df.drop_duplicates()
    return df


# Verifica que las variables tengan valores permitidos
def validar_valores(df):

    valores_validos = {
        "Diabetes_012": [0, 1, 2],
        "HighBP": [0, 1],
        "HighChol": [0, 1],
        "CholCheck": [0, 1],
        "Smoker": [0, 1],
        "Stroke": [0, 1],
        "HeartDiseaseorAttack": [0, 1],
        "PhysActivity": [0, 1],
        "Fruits": [0, 1],
        "Veggies": [0, 1],
        "HvyAlcoholConsump": [0, 1],
        "AnyHealthcare": [0, 1],
        "NoDocbcCost": [0, 1],
        "DiffWalk": [0, 1],
        "Sex": [0, 1],
        "GenHlth": [1, 2, 3, 4, 5],
        "Age": list(range(1, 14)),
        "Education": list(range(1, 7)),
        "Income": list(range(1, 9))
    }

    for columna, permitidos in valores_validos.items():

        encontrados = df[columna].unique()

        invalidos = [
            valor for valor in encontrados
            if valor not in permitidos
        ]

    return df


# Verifica valores fuera de los rangos permitidos
def validar_rangos(df):

    rangos = {
        "BMI": (10, 100),
        "MentHlth": (0, 30),
        "PhysHlth": (0, 30)
    }

    for columna, (minimo, maximo) in rangos.items():

        fuera_rango = df[
            (df[columna] < minimo) |
            (df[columna] > maximo)
        ]

    return df


# Calcula la distribución de diabetes
def analizar_distribucion_diabetes(df):

    conteo = (
        df["Diabetes_012"]
        .value_counts()
        .sort_index()
    )

    porcentaje = (
        df["Diabetes_012"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    return porcentaje


# Calcula el porcentaje de diabetes para una variable
def analizar_factor(df, columna):

    resultado = df.groupby(columna)["Diabetes_012"].apply(
        lambda x: (x == 2).mean() * 100
    )

    return resultado


# Compara los factores binarios y calcula sus razones
def comparar_factores(df):

    factores = [
        "HighBP",
        "HighChol",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "HvyAlcoholConsump",
        "DiffWalk"
    ]

    resultados = []

    for factor in factores:

        grupo_0 = df[df[factor] == 0]
        grupo_1 = df[df[factor] == 1]

        porcentaje_0 = (
            grupo_0["Diabetes_012"] == 2
        ).mean() * 100

        porcentaje_1 = (
            grupo_1["Diabetes_012"] == 2
        ).mean() * 100

        razon = porcentaje_1 / porcentaje_0

        resultados.append({
            "Factor": factor,
            "Diabetes sin factor (%)": porcentaje_0,
            "Diabetes con factor (%)": porcentaje_1,
            "Razón": razon
        })

    resultados = pd.DataFrame(resultados)

    resultados = resultados.sort_values(
        by="Razón",
        ascending=False
    )

    return resultados


# Agrupa el BMI y calcula diabetes por categoría
def analizar_bmi(df):

    grupos = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, float("inf")],
        labels=[
            "Bajo peso",
            "Normal",
            "Sobrepeso",
            "Obesidad"
        ]
    )

    resultado = df.groupby(
        grupos,
        observed=False
    )["Diabetes_012"].apply(
        lambda x: (x == 2).mean() * 100
    )

    return resultado


# Calcula el porcentaje de diabetes por grupo de edad
def analizar_edad(df):

    resultado = df.groupby(
        "Age"
    )["Diabetes_012"].apply(
        lambda x: (x == 2).mean() * 100
    )

    return resultado


# Compara la salud general con la presencia de diabetes
def analizar_salud_general(df):

    resultado = df.groupby(
        "GenHlth"
    )["Diabetes_012"].apply(
        lambda x: (x == 2).mean() * 100
    )

    return resultado


# Compara todas las razones y calcula el peso de cada factor
def comparar_razones(
    df_comparacion,
    resultado_bmi,
    resultado_edad,
    resultado_salud
):

    factores = []

    for _, fila in df_comparacion.iterrows():

        porcentaje_0 = fila["Diabetes sin factor (%)"]
        porcentaje_1 = fila["Diabetes con factor (%)"]

        if porcentaje_1 >= porcentaje_0:

            prevalencia_riesgo = porcentaje_1
            prevalencia_referencia = porcentaje_0
            condicion_riesgo = 1

        else:

            prevalencia_riesgo = porcentaje_0
            prevalencia_referencia = porcentaje_1
            condicion_riesgo = 0

        razon = (
            prevalencia_riesgo /
            prevalencia_referencia
        )

        factores.append({
            "Factor": fila["Factor"],
            "Condición de mayor riesgo": condicion_riesgo,
            "Razón": razon
        })

    bmi_referencia = resultado_bmi["Normal"]

    razon_bmi = (
        resultado_bmi.max() /
        bmi_referencia
    )

    factores.append({
        "Factor": "BMI",
        "Condición de mayor riesgo": resultado_bmi.idxmax(),
        "Razón": razon_bmi
    })

    edad_referencia = resultado_edad.loc[1]

    razon_edad = (
        resultado_edad.max() /
        edad_referencia
    )

    factores.append({
        "Factor": "Age",
        "Condición de mayor riesgo": resultado_edad.idxmax(),
        "Razón": razon_edad
    })

    salud_referencia = resultado_salud.loc[1]

    razon_salud = (
        resultado_salud.max() /
        salud_referencia
    )

    factores.append({
        "Factor": "GenHlth",
        "Condición de mayor riesgo": resultado_salud.idxmax(),
        "Razón": razon_salud
    })

    tabla = pd.DataFrame(factores)

    # Mide cuánto aumenta el riesgo respecto al valor neutro
    tabla["Fuerza"] = tabla["Razón"] - 1

    total_fuerza = tabla["Fuerza"].sum()

    # Convierte la fuerza estadística en pesos porcentuales
    tabla["Peso (%)"] = (
        tabla["Fuerza"] /
        total_fuerza
    ) * 100

    tabla = tabla.sort_values(
        by="Peso (%)",
        ascending=False
    )

    return tabla


# Calcula el puntaje de riesgo de un paciente
def calcular_puntaje(
    paciente,
    tabla_pesos,
    resultado_bmi,
    resultado_edad,
    resultado_salud
):

    puntaje = 0

    factores_binarios = [
        "HighBP",
        "HighChol",
        "Smoker",
        "Stroke",
        "HeartDiseaseorAttack",
        "PhysActivity",
        "HvyAlcoholConsump",
        "DiffWalk"
    ]

    # Suma los pesos de los factores binarios de riesgo
    for _, fila in tabla_pesos.iterrows():

        factor = fila["Factor"]

        if factor in factores_binarios:

            peso = fila["Peso (%)"]
            condicion = fila["Condición de mayor riesgo"]

            if paciente[factor] == condicion:
                puntaje += peso

    # Determina la categoría BMI del paciente
    if paciente["BMI"] < 18.5:
        categoria_bmi = "Bajo peso"

    elif paciente["BMI"] < 25:
        categoria_bmi = "Normal"

    elif paciente["BMI"] < 30:
        categoria_bmi = "Sobrepeso"

    else:
        categoria_bmi = "Obesidad"

    peso_bmi = tabla_pesos[
        tabla_pesos["Factor"] == "BMI"
    ]["Peso (%)"].iloc[0]

    razon_bmi = (
        resultado_bmi[categoria_bmi] /
        resultado_bmi["Normal"]
    )

    razon_bmi_max = (
        resultado_bmi.max() /
        resultado_bmi["Normal"]
    )

    # Asigna una parte del peso según el nivel de BMI
    if razon_bmi > 1 and razon_bmi_max > 1:

        puntaje += peso_bmi * (
            (razon_bmi - 1) /
            (razon_bmi_max - 1)
        )

    peso_edad = tabla_pesos[
        tabla_pesos["Factor"] == "Age"
    ]["Peso (%)"].iloc[0]

    razon_edad = (
        resultado_edad.loc[paciente["Age"]] /
        resultado_edad.loc[1]
    )

    razon_edad_max = (
        resultado_edad.max() /
        resultado_edad.loc[1]
    )

    # Asigna una parte del peso según el grupo de edad
    if razon_edad > 1 and razon_edad_max > 1:

        puntaje += peso_edad * (
            (razon_edad - 1) /
            (razon_edad_max - 1)
        )

    peso_salud = tabla_pesos[
        tabla_pesos["Factor"] == "GenHlth"
    ]["Peso (%)"].iloc[0]

    razon_salud = (
        resultado_salud.loc[paciente["GenHlth"]] /
        resultado_salud.loc[1]
    )

    razon_salud_max = (
        resultado_salud.max() /
        resultado_salud.loc[1]
    )

    # Asigna una parte del peso según la salud general
    if razon_salud > 1 and razon_salud_max > 1:

        puntaje += peso_salud * (
            (razon_salud - 1) /
            (razon_salud_max - 1)
        )

    return round(puntaje, 2)


# Calcula el puntaje de todos los pacientes
def calcular_puntajes_dataset(
    df,
    tabla_pesos,
    resultado_bmi,
    resultado_edad,
    resultado_salud
):

    puntajes = []

    for _, fila in df.iterrows():

        paciente = fila.to_dict()

        puntaje = calcular_puntaje(
            paciente,
            tabla_pesos,
            resultado_bmi,
            resultado_edad,
            resultado_salud
        )

        puntajes.append(puntaje)

    df_resultado = df.copy()

    df_resultado["Puntaje"] = puntajes

    return df_resultado


# Calcula los límites estadísticos para los niveles de riesgo
def calcular_limites_riesgo(df_resultado):

    limite_bajo = df_resultado["Puntaje"].quantile(0.33)
    limite_alto = df_resultado["Puntaje"].quantile(0.66)

    return limite_bajo, limite_alto


# Convierte el puntaje en riesgo bajo, medio o alto
def clasificar_riesgo(
    puntaje,
    limite_bajo,
    limite_alto
):

    if puntaje <= limite_bajo:
        return "Bajo"

    elif puntaje <= limite_alto:
        return "Medio"

    else:
        return "Alto"


# Clasifica todos los pacientes según su puntaje
def clasificar_dataset(
    df_resultado,
    limite_bajo,
    limite_alto
):

    df_resultado["Riesgo"] = df_resultado["Puntaje"].apply(
        lambda puntaje: clasificar_riesgo(
            puntaje,
            limite_bajo,
            limite_alto
        )
    )

    return df_resultado


# Calcula qué porcentaje real de cada nivel tiene diabetes
def validar_clasificacion(df_resultado):

    resultado = df_resultado.groupby(
        "Riesgo"
    )["Diabetes_012"].apply(
        lambda x: (x == 2).mean() * 100
    )

    resultado = resultado.reindex(
        ["Bajo", "Medio", "Alto"]
    )

    return resultado


# Ruta del dataset
ruta = r"C:\Users\Asus Vivobook\Downloads\diabetes_012_health_indicators_BRFSS2015.csv"

# Carga y valida los datos
df = cargar_datos(ruta)

df = validar_nulos(df)

df = validar_duplicados(df)

df = validar_valores(df)

df = validar_rangos(df)


# Toma una muestra pequeña para trabajar más rápido
df_muestra = df.sample(
    n=5000,
    random_state=42
)


# Realiza los análisis estadísticos sobre la muestra
distribucion = analizar_distribucion_diabetes(df_muestra)

df_comparacion = comparar_factores(df_muestra)

resultado_bmi = analizar_bmi(df_muestra)

resultado_edad = analizar_edad(df_muestra)

resultado_salud = analizar_salud_general(df_muestra)


# Calcula los pesos estadísticos
tabla_pesos = comparar_razones(
    df_comparacion,
    resultado_bmi,
    resultado_edad,
    resultado_salud
)


# Calcula el puntaje de los pacientes de la muestra
df_resultado = calcular_puntajes_dataset(
    df_muestra,
    tabla_pesos,
    resultado_bmi,
    resultado_edad,
    resultado_salud
)


# Calcula los límites de riesgo
limite_bajo, limite_alto = calcular_limites_riesgo(
    df_resultado
)


# Clasifica los pacientes
df_resultado = clasificar_dataset(
    df_resultado,
    limite_bajo,
    limite_alto
)


# Valida la clasificación
resultado_validacion = validar_clasificacion(
    df_resultado
)

# Calcula el IMC de forma automática a partir del peso y la altura
def calcular_bmi_automatico():
    print("\n--- Cálculo de Índice de Masa Corporal (IMC) ---")
    
    while True:
        try:
            peso = float(input("Ingrese su peso en kilogramos (ej. 75): "))
            altura = float(input("Ingrese su altura en metros (ej. 1.75): "))
            
            if peso <= 0 or altura <= 0:
                print("Por favor, ingrese valores mayores a cero.")
                continue
                
            bmi = peso / (altura ** 2)
            print(f"IMC calculado: {bmi:.2f}")
            return bmi
        except ValueError:
            print("Entrada inválida. Por favor, use números válidos.")


# Traduce la edad real del paciente al formato numérico requerido por el modelo (1 al 13)
def calcular_grupo_edad():
    print("\n--- Selección de Grupo de Edad ---")
    print("1: 18 - 24 años")
    print("2: 25 - 29 años")
    print("3: 30 - 34 años")
    print("4: 35 - 39 años")
    print("5: 40 - 44 años")
    print("6: 45 - 49 años")
    print("7: 50 - 54 años")
    print("8: 55 - 59 años")
    print("9: 60 - 64 años")
    print("10: 65 - 69 años")
    print("11: 70 - 74 años")
    print("12: 75 - 79 años")
    print("13: 80 años o más")

    while True:
        try:
            opcion = int(input("Seleccione el número correspondiente a su rango de edad (1-13): "))
            if 1 <= opcion <= 13:
                return opcion
            print("Opción fuera de rango. Elija un número entre 1 y 13.")
        except ValueError:
            print("Entrada inválida. Ingrese un número entero.")


# Solicita de forma guiada y amigable los datos del paciente
def ingresar_paciente():
    print("\n========================================")
    print("   EVALUADOR CLÍNICO DE RIESGO DE DIABETES")
    print("========================================")

    paciente = {}

    # Validadores binarios con formato claro
    def preguntar_s_n(pregunta):
        while True:
            resp = input(f"{pregunta} (s/n): ").strip().lower()
            if resp in ['s', 'si', 'sí', '1']:
                return 1
            elif resp in ['n', 'no', '0']:
                return 0
            print("Por favor,responda 's' para Sí o 'n' para No.")

    paciente["HighBP"] = preguntar_s_n("¿Padece de presión arterial alta?")
    paciente["HighChol"] = preguntar_s_n("¿Tiene niveles de colesterol alto?")
    paciente["Smoker"] = preguntar_s_n("¿Ha fumado al menos 100 cigarrillos en su vida?")
    paciente["Stroke"] = preguntar_s_n("¿Ha sufrido algún derrame cerebral (ictus)?")
    paciente["HeartDiseaseorAttack"] = preguntar_s_n("¿Tiene antecedentes de enfermedad cardíaca o infarto?")
    paciente["PhysActivity"] = preguntar_s_n("¿Realiza actividad física o ejercicio de forma regular?")
    paciente["HvyAlcoholConsump"] = preguntar_s_n("¿Consume alcohol en exceso (hombres >14 tragos/sem, mujeres >7)?")
    paciente["DiffWalk"] = preguntar_s_n("¿Tiene dificultad seria para caminar o subir escaleras?")

    # Cálculo automático de BMI y edad guiada
    paciente["BMI"] = calcular_bmi_automatico()
    paciente["Age"] = calcular_grupo_edad()

    print("\n--- Evaluación de Salud General ---")
    print("1: Excelente | 2: Muy buena | 3: Buena | 4: Regular | 5: Mala")
    while True:
        try:
            gen_hlth = int(input("¿Cómo calificaría su estado de salud general (1-5)?: "))
            if 1 <= gen_hlth <= 5:
                paciente["GenHlth"] = gen_hlth
                break
            print("Ingrese un valor entre 1 y 5.")
        except ValueError:
            print("Entrada inválida.")

    return paciente


# Calcula y muestra el riesgo del nuevo paciente con un reporte formateado
def evaluar_nuevo_paciente(
    tabla_pesos,
    resultado_bmi,
    resultado_edad,
    resultado_salud,
    limite_bajo,
    limite_alto
):

    paciente = ingresar_paciente()

    puntaje = calcular_puntaje(
        paciente,
        tabla_pesos,
        resultado_bmi,
        resultado_edad,
        resultado_salud
    )

    riesgo = clasificar_riesgo(
        puntaje,
        limite_bajo,
        limite_alto
    )

    print("\n========================================")
    print("         INFORME DE EVALUACIÓN")
    print("========================================")
    print(f" Puntaje de Riesgo Calculado : {puntaje}")
    print(f" Categoría de Riesgo         : {riesgo.upper()}")
    print("========================================")
    
    if riesgo == "Bajo":
        print(" Recomendación: Mantenga sus hábitos de vida saludables y chequeos preventivos anuales.")
    elif riesgo == "Medio":
        print(" Recomendación: Riesgo moderado (Prediabetes). Se sugiere una evaluación analítica de glucosa con su médico.")
    else:
        print(" Recomendación: Alto riesgo detectado. Se recomienda consultar prioritariamente con un profesional de salud.")

    return paciente, puntaje, riesgo


# Ejecución de la interfaz interactiva
evaluar_nuevo_paciente(
    tabla_pesos,
    resultado_bmi,
    resultado_edad,
    resultado_salud,
    limite_bajo,
    limite_alto
)

# Muestra la distribución de pacientes según su condición de diabetes
def graficar_distribucion_diabetes(df):

    conteo = df["Diabetes_012"].value_counts().sort_index()

    etiquetas = [
        "Sin diabetes",
        "Prediabetes",
        "Diabetes"
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        etiquetas,
        conteo.values
    )

    plt.title("Distribución de pacientes según condición de diabetes")
    plt.xlabel("Condición")
    plt.ylabel("Cantidad de pacientes")

    plt.tight_layout()
    plt.savefig("grafica_distribucion_diabetes.png", dpi=300)
    plt.show()


# Compara la prevalencia de diabetes con y sin cada factor clínico
def graficar_factores_clinicos(df_comparacion):

    tabla = df_comparacion.copy()

    posiciones = range(len(tabla))

    plt.figure(figsize=(11, 6))

    plt.bar(
        [x - 0.2 for x in posiciones],
        tabla["Diabetes sin factor (%)"],
        width=0.4,
        label="Sin factor"
    )

    plt.bar(
        [x + 0.2 for x in posiciones],
        tabla["Diabetes con factor (%)"],
        width=0.4,
        label="Con factor"
    )

    plt.xticks(
        posiciones,
        tabla["Factor"],
        rotation=45,
        ha="right"
    )

    plt.title("Prevalencia de diabetes según factores clínicos")
    plt.xlabel("Factor clínico")
    plt.ylabel("Pacientes con diabetes (%)")

    plt.legend()

    plt.tight_layout()
    plt.savefig("grafica_factores_clinicos.png", dpi=300)
    plt.show()


# Muestra el peso estadístico asignado a cada factor
def graficar_pesos_estadisticos(tabla_pesos):

    tabla = tabla_pesos.sort_values(
        "Peso (%)",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        tabla["Factor"],
        tabla["Peso (%)"]
    )

    plt.title("Peso estadístico de los factores en el puntaje de riesgo")
    plt.xlabel("Peso dentro del puntaje (%)")
    plt.ylabel("Factor")

    plt.tight_layout()
    plt.savefig("grafica_pesos_estadisticos.png", dpi=300)
    plt.show()


# Muestra la prevalencia real de diabetes en cada nivel de riesgo
def graficar_validacion_riesgo(resultado_validacion):

    plt.figure(figsize=(8, 5))

    plt.bar(
        resultado_validacion.index,
        resultado_validacion.values
    )

    plt.title("Prevalencia de diabetes según nivel de riesgo calculado")
    plt.xlabel("Nivel de riesgo")
    plt.ylabel("Pacientes con diabetes (%)")

    plt.tight_layout()
    plt.savefig("grafica_validacion_riesgo.png", dpi=300)
    plt.show()
    
# Genera las gráficas del análisis
graficar_distribucion_diabetes(
    df_muestra
)

graficar_factores_clinicos(
    df_comparacion
)

graficar_pesos_estadisticos(
    tabla_pesos
)

graficar_validacion_riesgo(
    resultado_validacion
)

 
