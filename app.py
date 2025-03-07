# Derivados-
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

#######################
# Configuración de la página
st.set_page_config(
    page_title="Modelos de Valoración de Opciones",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado 
st.markdown("""
<style>
/* Ajustar el tamaño y la alineación de los contenedores de valores CALL y PUT */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Ajustar el padding para controlar la altura */
    width: auto; /* Ancho automático para responsividad, o fijar un ancho si es necesario */
    margin: 0 auto; /* Centrar el contenedor */
}

/* Clases personalizadas para los valores CALL y PUT */
.metric-call {
    background-color: #90ee90; /* Fondo verde claro */
    color: black; /* Color de fuente negro */
    margin-right: 10px; /* Espaciado entre CALL y PUT */
    border-radius: 10px; /* Esquinas redondeadas */
}

.metric-put {
    background-color: #ffcccb; /* Fondo rojo claro */
    color: black; /* Color de fuente negro */
    border-radius: 10px; /* Esquinas redondeadas */
}

/* Estilo para el texto del valor */
.metric-value {
    font-size: 1.5rem; /* Ajustar el tamaño de la fuente */
    font-weight: bold;
    margin: 0; /* Eliminar márgenes por defecto */
}

/* Estilo para el texto de la etiqueta */
.metric-label {
    font-size: 1rem; /* Ajustar el tamaño de la fuente */
    margin-bottom: 4px; /* Espaciado entre la etiqueta y el valor */
}

</style>
""", unsafe_allow_html=True)

#  BlackScholes
class BlackScholes:
    def __init__(
        self,
        tiempo_hasta_vencimiento: float,
        precio_ejercicio: float,
        precio_actual: float,
        volatilidad: float,
        tasa_interes: float,
    ):
        self.tiempo_hasta_vencimiento = tiempo_hasta_vencimiento
        self.precio_ejercicio = precio_ejercicio
        self.precio_actual = precio_actual
        self.volatilidad = volatilidad
        self.tasa_interes = tasa_interes

    def calcular_precios(self):
        tiempo_hasta_vencimiento = self.tiempo_hasta_vencimiento
        precio_ejercicio = self.precio_ejercicio
        precio_actual = self.precio_actual
        volatilidad = self.volatilidad
        tasa_interes = self.tasa_interes

        d1 = (
            np.log(precio_actual / precio_ejercicio) +
            (tasa_interes + 0.5 * volatilidad ** 2) * tiempo_hasta_vencimiento
        ) / (
            volatilidad * np.sqrt(tiempo_hasta_vencimiento)
        )
        d2 = d1 - volatilidad * np.sqrt(tiempo_hasta_vencimiento)

        precio_call = precio_actual * norm.cdf(d1) - (
            precio_ejercicio * np.exp(-(tasa_interes * tiempo_hasta_vencimiento)) * norm.cdf(d2)
        )
        precio_put = (
            precio_ejercicio * np.exp(-(tasa_interes * tiempo_hasta_vencimiento)) * norm.cdf(-d2)
        ) - precio_actual * norm.cdf(-d1)

        self.precio_call = precio_call
        self.precio_put = precio_put

        # Griegas
        # Delta
        self.delta_call = norm.cdf(d1)
        self.delta_put = 1 - norm.cdf(d1)

        # Gamma
        self.gamma_call = norm.pdf(d1) / (
            precio_ejercicio * volatilidad * np.sqrt(tiempo_hasta_vencimiento)
        )
        self.gamma_put = self.gamma_call

        return precio_call, precio_put

# Modelo Heston
class Heston:
    def __init__(
        self,
        S0: float,
        v0: float,
        r: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        T: float,
        N: int,
        M: int,
    ):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.T = T
        self.N = N
        self.M = M

    def simular_trayectorias(self):
        dt = self.T / self.N
        S = np.zeros((self.N, self.M))
        v = np.zeros((self.N, self.M))
        
        S[0, :] = self.S0
        v[0, :] = self.v0

        for t in range(1, self.N):
            Z1 = np.random.normal(0, 1, self.M)
            Z2 = np.random.normal(0, 1, self.M)
            W1 = Z1
            W2 = self.rho * Z1 + np.sqrt(1 - self.rho**2) * Z2
            
            v[t, :] = np.abs(v[t-1, :] + self.kappa * (self.theta - v[t-1, :]) * dt + 
                             self.sigma * np.sqrt(v[t-1, :]) * np.sqrt(dt) * W2)
            S[t, :] = S[t-1, :] * np.exp((self.r - 0.5 * v[t-1, :]) * dt + 
                       np.sqrt(v[t-1, :]) * np.sqrt(dt) * W1)

        return S, v

    def calcular_precio_opcion(self, K, tipo="call"):
        S, _ = self.simular_trayectorias()
        payoff = np.maximum(S[-1, :] - K, 0) if tipo == "call" else np.maximum(K - S[-1, :], 0)
        precio = np.exp(-self.r * self.T) * np.mean(payoff)
        return precio

# Barra lateral para entradas del usuario
with st.sidebar:
    st.title("📊 Modelos de Valoración de Opciones")
    st.write("`Creado por:`")
    linkedin_url = "https://www.linkedin.com/in/juan-leonardo-pati%C3%B1o-martinez-a39032137/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Juan Leonardo Patiño Martínez`</a>', unsafe_allow_html=True)

    # Parámetros para Black-Scholes
    st.markdown("### Parámetros para Black-Scholes")
    precio_actual = st.number_input("Precio Actual del Activo", value=100.0)
    precio_ejercicio = st.number_input("Precio de Ejercicio", value=100.0)
    tiempo_hasta_vencimiento = st.number_input("Tiempo hasta el Vencimiento (Años)", value=1.0)
    volatilidad = st.number_input("Volatilidad (σ)", value=0.2)
    tasa_interes = st.number_input("Tasa de Interés Libre de Riesgo", value=0.05)

    # Parámetros para Heston
    st.markdown("### Parámetros para el Modelo de Heston")
    S0 = st.number_input("Precio Inicial del Activo (S0)", value=100.0)
    v0 = st.number_input("Volatilidad Inicial (v0)", value=0.04)
    kappa = st.number_input("Velocidad de Reversión (kappa)", value=2.0)
    theta = st.number_input("Media de Largo Plazo de la Volatilidad (theta)", value=0.04)
    sigma = st.number_input("Volatilidad de la Volatilidad (sigma)", value=0.3)
    rho = st.slider("Correlación entre el Precio y la Volatilidad (rho)", min_value=-1.0, max_value=1.0, value=-0.7)
    T = st.number_input("Tiempo en Años (T)", value=1.0)
    N = st.number_input("Número de Pasos en la Simulación (N)", value=252)
    M = st.number_input("Número de Trayectorias a Simular (M)", value=1000)

# Página principal para mostrar los resultados
st.title("Modelos de Valoración de Opciones")

# Black-Scholes
st.markdown("## Modelo Black-Scholes")
bs_model = BlackScholes(tiempo_hasta_vencimiento, precio_ejercicio, precio_actual, volatilidad, tasa_interes)
precio_call, precio_put = bs_model.calcular_precios()

# Mostrar valores CALL y PUT en tablas coloreadas
col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Valor CALL</div>
                <div class="metric-value">${precio_call:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Valor PUT</div>
                <div class="metric-value">${precio_put:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Modelo de Heston
st.markdown("## Modelo de Heston")
heston_model = Heston(S0, v0, tasa_interes, kappa, theta, sigma, rho, T, N, M)

# Simular trayectorias
S_simulado, v_simulado = heston_model.simular_trayectorias()

# Graficar las trayectorias del precio
st.markdown("### Simulación de Trayectorias del Precio con el Modelo de Heston")
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(min(M, 10)):  # Mostrar solo 10 trayectorias para claridad
    ax.plot(S_simulado[:, i], alpha=0.7)
ax.set_xlabel("Tiempo")
ax.set_ylabel("Precio del Activo")
ax.set_title("Simulación del Modelo de Heston")
st.pyplot(fig)

# Calcular precios de opciones con Heston
st.markdown("### Precios de Opciones con el Modelo de Heston")
precio_call_heston = heston_model.calcular_precio_opcion(precio_ejercicio, tipo="call")
precio_put_heston = heston_model.calcular_precio_opcion(precio_ejercicio, tipo="put")

col1, col2 = st.columns([1, 1], gap="small")

with col1:
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Valor CALL (Heston)</div>
                <div class="metric-value">${precio_call_heston:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Valor PUT (Heston)</div>
                <div class="metric-value">${precio_put_heston:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
