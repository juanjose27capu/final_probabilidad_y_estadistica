import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from scipy import stats

# ===========================================================================
# 1. LÓGICA ESTADÍSTICA (Separada de la interfaz gráfica)
# ===========================================================================

def media_muestral(x):
    return np.mean(np.asarray(x, dtype=float))

def varianza_muestral(x):
    return np.var(np.asarray(x, dtype=float), ddof=1)

def desvio_muestral(x):
    return np.sqrt(varianza_muestral(x))

def coef_variacion(x):
    m = media_muestral(x)
    if m == 0:
        return np.nan
    return desvio_muestral(x) / m

def sturges(n):
    n_bruto = 1 + 3.3 * np.log10(n)
    return int(np.ceil(n_bruto))

# Configuraciones globales estadísticas
NIVEL_CONFIANZA = 0.95
ALFA = 1 - NIVEL_CONFIANZA
NOTA_APROBACION = 10.0

# ===========================================================================
# 2. CARGA Y PREPARACIÓN DE DATOS
# ===========================================================================

class DatosEstadisticos:
    def __init__(self):
        carpeta = Path(__file__).resolve().parent
        ruta_csv = carpeta / "muestra-estudiantes-matematica.csv"
        
        try:
            self.df = pd.read_csv(ruta_csv)
            print(f"Dataset cargado correctamente: n = {len(self.df)} estudiantes.")
        except Exception as e:
            raise FileNotFoundError(f"Error al cargar el CSV. Asegurate de que esté en {carpeta}\nDetalle: {e}")

        self.df_trabajo = self.df.copy()
        
        # Casteo de la variable binaria a int
        if 'en_pareja' in self.df_trabajo.columns:
            self.df_trabajo['en_pareja'] = self.df_trabajo['en_pareja'].astype(int)

# ===========================================================================
# 3. INTERFAZ GRÁFICA (Tkinter + Matplotlib)
# ===========================================================================

class PresentacionEstadistica(tk.Tk):
    def __init__(self, datos):
        super().__init__()
        self.datos = datos
        self.df = datos.df_trabajo
        
        self.title("Probabilidad y Estadística — Rendimiento en Matemáticas")
        self.geometry("1100x750")
        self.configure(bg="#F0F0F0")
        
        self.unidades = ["Unidad 6", "Unidad 7", "Unidad 8", "Unidad 9", "Unidad 10"]
        self.indice_actual = 0
        
        self._construir_interfaz()
        self.cargar_unidad(0)

    def _construir_interfaz(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#F0F0F0")
        
        # --- PANEL IZQUIERDO (NAVEGACIÓN) ---
        frame_nav = tk.Frame(self, bg="#2C3E50", width=200)
        frame_nav.pack(side=tk.LEFT, fill=tk.Y)
        
        titulo_nav = tk.Label(frame_nav, text="UNIDADES", bg="#2C3E50", fg="white", font=("Segoe UI", 14, "bold"))
        titulo_nav.pack(pady=20)
        
        self.botones_nav = []
        nombres_botones = ["Unidad 6 — EDA", "Unidad 7 — Est. Puntual", "Unidad 8 — Int. Confianza", 
                           "Unidad 9 — Hipótesis", "Unidad 10 — Regresión"]
        
        for i, nombre in enumerate(nombres_botones):
            btn = tk.Button(frame_nav, text=nombre, bg="#34495E", fg="white", font=("Segoe UI", 11),
                            relief=tk.FLAT, activebackground="#1ABC9C", activeforeground="white",
                            command=lambda idx=i: self.cargar_unidad(idx))
            btn.pack(fill=tk.X, padx=10, pady=5)
            self.botones_nav.append(btn)
            
        frame_controles = tk.Frame(frame_nav, bg="#2C3E50")
        frame_controles.pack(side=tk.BOTTOM, fill=tk.X, pady=20)
        
        tk.Button(frame_controles, text="← Anterior", bg="#1ABC9C", fg="white", relief=tk.FLAT,
                  command=self.unidad_anterior).pack(side=tk.LEFT, padx=10, expand=True, fill=tk.X)
        tk.Button(frame_controles, text="Siguiente →", bg="#1ABC9C", fg="white", relief=tk.FLAT,
                  command=self.unidad_siguiente).pack(side=tk.RIGHT, padx=10, expand=True, fill=tk.X)

        tk.Button(frame_controles, text="💾 Guardar Gráfico", bg="#E67E22", fg="white", 
                  font=("Segoe UI", 10, "bold"), relief=tk.FLAT,
                  command=self.guardar_grafico).pack(side=tk.BOTTOM, padx=10, pady=(15, 0), fill=tk.X)
        
        # --- PANEL DERECHO (CONTENIDO) ---
        frame_main = tk.Frame(self, bg="#F0F0F0")
        frame_main.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        self.lbl_titulo = ttk.Label(frame_main, text="", style="Title.TLabel")
        self.lbl_titulo.pack(anchor=tk.W, pady=(0, 10))
        
        self.fig = plt.Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        frame_texto = tk.Frame(frame_main)
        frame_texto.pack(fill=tk.BOTH, expand=False, pady=10)
        
        scroll = tk.Scrollbar(frame_texto)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_area = tk.Text(frame_texto, height=12, wrap=tk.WORD, yscrollcommand=scroll.set,
                                 font=("Segoe UI", 11), bg="#FAFAFA", relief=tk.SOLID, bd=1)
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.config(command=self.text_area.yview)
        
        self.text_area.tag_configure("header", font=("Segoe UI", 12, "bold"), foreground="#2980B9")

    def _actualizar_texto(self, concepto, explicacion, formula, resultado, interpretacion):
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete("1.0", tk.END)
        
        if concepto:
            self.text_area.insert(tk.END, "Concepto\n", "header")
            self.text_area.insert(tk.END, f"{concepto}\n\n")
            
        self.text_area.insert(tk.END, "Explicación\n", "header")
        self.text_area.insert(tk.END, f"{explicacion}\n\n")
        
        if formula:
            self.text_area.insert(tk.END, "Fórmula\n", "header")
            self.text_area.insert(tk.END, f"{formula}\n\n")
            
        self.text_area.insert(tk.END, "Resultado\n", "header")
        self.text_area.insert(tk.END, f"{resultado}\n\n")
        
        self.text_area.insert(tk.END, "Interpretación\n", "header")
        self.text_area.insert(tk.END, f"{interpretacion}\n")
        
        self.text_area.config(state=tk.DISABLED)

    def resaltar_boton(self, indice):
        for i, btn in enumerate(self.botones_nav):
            if i == indice:
                btn.config(bg="#1ABC9C", font=("Segoe UI", 11, "bold"))
            else:
                btn.config(bg="#34495E", font=("Segoe UI", 11, "normal"))

    def unidad_anterior(self):
        if self.indice_actual > 0:
            self.cargar_unidad(self.indice_actual - 1)

    def unidad_siguiente(self):
        if self.indice_actual < len(self.unidades) - 1:
            self.cargar_unidad(self.indice_actual + 1)

    def cargar_unidad(self, indice):
        self.indice_actual = indice
        self.resaltar_boton(indice)
        self.fig.clear()
        
        if indice == 0:
            self.mostrar_u6()
        elif indice == 1:
            self.mostrar_u7()
        elif indice == 2:
            self.mostrar_u8()
        elif indice == 3:
            self.mostrar_u9()
        elif indice == 4:
            self.mostrar_u10()
            
        self.canvas.draw()

    # ===========================================================================
    # ANÁLISIS DE UNIDADES
    # ===========================================================================

    def mostrar_u6(self):
            self.lbl_titulo.config(text="Unidad 6 — Análisis Exploratorio de Datos (EDA)")
            notas = self.df["nota_final"].dropna()
            n = len(notas)
            media = media_muestral(notas)
            desvio = desvio_muestral(notas)
            cv = coef_variacion(notas)
            k = sturges(n)

            # 1. Histograma (Arriba - Izquierda)
            ax1 = self.fig.add_subplot(221)
            ax1.hist(notas, bins=k, color="#4A5A7B", edgecolor="black")
            ax1.axvline(media, color="#E3655B", linestyle="--", label=f"Media = {media:.2f}")
            ax1.set_title(r"Distribución Notas Finales ($\bar{x}$)")
            ax1.set_xlabel("Nota (0 a 20)")
            ax1.set_ylabel("Frecuencia")
            ax1.legend()

            # 2. Gráfico de Torta: En Pareja (Arriba - Derecha)
            ax2 = self.fig.add_subplot(222)
            pareja_counts = self.df['en_pareja'].value_counts()
            # Mapeo según tus reglas: 0 = Sí, 1 = No
            labels_pareja = ['Sí' if val == 0 else 'No' for val in pareja_counts.index]
            colores_torta = ['#FF9999', '#66B3FF'] # Salmón y celeste como en tu imagen
            ax2.pie(pareja_counts, labels=labels_pareja, autopct='%1.1f%%', colors=colores_torta, startangle=90)
            ax2.set_title("Estado Sentimental (En pareja)")

            # Mapeo de niveles educativos para los gráficos de barras
            mapa_edu = {0: 'Ninguna', 1: 'Primaria', 2: 'Media', 3: 'Secundaria', 4: 'Superior'}
            colores_barras = ['#4A5A7B', '#2D827A', '#6CB468', '#ECA400', '#E3655B']

            # 3. Gráfico de Barras: Educación Madre (Abajo - Izquierda)
            ax3 = self.fig.add_subplot(223)
            edu_m_counts = self.df['educacion_madre'].value_counts().sort_index()
            etiquetas_m = [mapa_edu.get(x, str(x)) for x in edu_m_counts.index]
            ax3.bar(etiquetas_m, edu_m_counts.values, color=colores_barras[:len(edu_m_counts)])
            ax3.set_title("Nivel Educativo: Madre")
            ax3.tick_params(axis='x', rotation=15, labelsize=8)

            # 4. Gráfico de Barras: Educación Padre (Abajo - Derecha)
            ax4 = self.fig.add_subplot(224)
            edu_p_counts = self.df['educacion_padre'].value_counts().sort_index()
            etiquetas_p = [mapa_edu.get(x, str(x)) for x in edu_p_counts.index]
            ax4.bar(etiquetas_p, edu_p_counts.values, color=colores_barras[:len(edu_p_counts)])
            ax4.set_title("Nivel Educativo: Padre")
            ax4.tick_params(axis='x', rotation=15, labelsize=8)

            self.fig.tight_layout()

            # Textos actualizados para la interfaz
            concepto = "Distribución Cuantitativa y Categórica del Perfil Estudiantil."
            explicacion = (
                "El Análisis Exploratorio de Datos (EDA) abarca tanto el rendimiento académico como el "
                "perfil demográfico de la muestra:\n"
                "• Histograma: Distribución de la Nota Final (G3).\n"
                "• Torta: Proporción de estudiantes en una relación de pareja.\n"
                "• Barras: Nivel educativo alcanzado por madres y padres."
            )
            formula = "Media: x̄ = Σxi / n\nCV = (S / x̄) × 100"
            resultado = f"Nota media (x̄) = {media:.2f} | Desvío (S) = {desvio:.2f} | CV = {cv:.3f}"
            interpretacion = (
                "Los gráficos categóricos complementan el análisis de rendimiento permitiendo observar el "
                "perfil social de los estudiantes. Esto es clave ya que factores como la educación parental o el "
                "estado sentimental pueden ser variables ocultas que influyan en el desempeño en matemáticas."
            )
            
            self._actualizar_texto(concepto, explicacion, formula, resultado, interpretacion)

    def mostrar_u7(self):
        self.lbl_titulo.config(text="Unidad 7 — Estimación Puntual")
        y_int = self.df["nota_final"].dropna().to_numpy()
        n_int = len(y_int)
        xbarra_int = media_muestral(y_int)
        s2_int = varianza_muestral(y_int)
        s_int = desvio_muestral(y_int)

        ax = self.fig.add_subplot(111)
        ax.boxplot(y_int, vert=False, patch_artist=True, boxprops=dict(facecolor="#9b59b6"))
        ax.plot(xbarra_int, 1, 'rD', markersize=8, label=f'Media Estimada: {xbarra_int:.2f}')
        ax.set_title(r"Estimación Puntual de $\mu$ (Nota Final)")
        ax.set_xlabel("Nota (0 a 20)")
        ax.set_yticks([])
        ax.legend()
        self.fig.tight_layout()

        concepto = "Estimadores Puntuales (Media y Varianza Poblacional)"
        explicacion = "Utilizamos los datos de la muestra para estimar parámetros desconocidos de la población total de estudiantes de las escuelas evaluadas."
        formula = "Estimador de μ: x̄ = Σxi / n\nEstimador de σ²: S² = Σ(xi - x̄)² / (n - 1)"
        resultado = f"Tamaño (n) = {n_int}\nx̄ = {xbarra_int:.4f}\nS² = {s2_int:.4f}\nS = {s_int:.4f}"
        interpretacion = f"Nuestra mejor aproximación al promedio poblacional real de la nota final es {xbarra_int:.2f}. La dispersión estimada en la población es de {s_int:.2f} puntos. Se utiliza n-1 en el cálculo de S² para asegurar un estimador insesgado."

        self._actualizar_texto(concepto, explicacion, formula, resultado, interpretacion)

    def mostrar_u8(self):
        self.lbl_titulo.config(text="Unidad 8 — Estimación por Intervalos")
        y_int = self.df["nota_final"].dropna().to_numpy()
        n_int = len(y_int)
        xbarra_int = media_muestral(y_int)
        s2_int = varianza_muestral(y_int)
        s_int = desvio_muestral(y_int)
        
        gl = n_int - 1
        t_crit = stats.t.ppf(1 - ALFA / 2, df=gl)
        error_est = s_int / np.sqrt(n_int)
        margen = t_crit * error_est
        ic_mu = (xbarra_int - margen, xbarra_int + margen)

        chi2_der = stats.chi2.ppf(1 - ALFA / 2, df=gl)
        chi2_izq = stats.chi2.ppf(ALFA / 2, df=gl)
        ic_sigma2 = ((gl * s2_int) / chi2_der, (gl * s2_int) / chi2_izq)

        ax1 = self.fig.add_subplot(121)
        ax1.errorbar(x=1, y=xbarra_int, yerr=[[xbarra_int - ic_mu[0]], [ic_mu[1] - xbarra_int]],
                     fmt="o", color="#4C72B0", ecolor="#C44E52", capsize=8, markersize=8)
        ax1.axhline(NOTA_APROBACION, color="gray", linestyle=":", label="Nota de Aprobación (10)")
        ax1.set_xlim(0.5, 1.5)
        ax1.set_xticks([])
        ax1.set_title(rf"IC 95% para $\mu$: [{ic_mu[0]:.2f}, {ic_mu[1]:.2f}]")
        ax1.legend()

        ax2 = self.fig.add_subplot(122)
        ax2.errorbar(x=1, y=s2_int, yerr=[[s2_int - ic_sigma2[0]], [ic_sigma2[1] - s2_int]],
                     fmt="s", color="#55A868", ecolor="#DD8452", capsize=8, markersize=8)
        ax2.set_xlim(0.5, 1.5)
        ax2.set_xticks([])
        ax2.set_title(rf"IC 95% para $\sigma^2$: [{ic_sigma2[0]:.2f}, {ic_sigma2[1]:.2f}]")
        self.fig.tight_layout()

        concepto = "Intervalos de Confianza para la Media (μ) y Varianza (σ²)"
        explicacion = "Calculamos un rango de valores en el cual, con una confianza del 95%, se encuentran la nota media y su dispersión a nivel poblacional."
        formula = r"IC(μ): x̄ ± t_{α/2, n-1} * (S / √n)"
        resultado = f"IC 95% para μ: [{ic_mu[0]:.4f} ; {ic_mu[1]:.4f}]\nIC 95% para σ²: [{ic_sigma2[0]:.4f} ; {ic_sigma2[1]:.4f}]"
        interpretacion = "Si repitiéramos el muestreo múltiples veces, el 95% de los intervalos contendrían el parámetro poblacional real. Al encontrarse todo el intervalo de confianza para la media por encima de 10, hay fuertes indicios de una tendencia general a la aprobación."

        self._actualizar_texto(concepto, explicacion, formula, resultado, interpretacion)

    def mostrar_u9(self):
        self.lbl_titulo.config(text="Unidad 9 — Pruebas de Hipótesis")
        y_int = self.df["nota_final"].dropna().to_numpy()
        n_int = len(y_int)
        
        t_obs, p_unilateral = stats.ttest_1samp(y_int, popmean=NOTA_APROBACION, alternative="greater")
        gl = n_int - 1
        t_crit_uni = stats.t.ppf(1 - ALFA, df=gl)

        ax = self.fig.add_subplot(111)
        x_t = np.linspace(-4, 5, 400)
        ax.plot(x_t, stats.t.pdf(x_t, df=gl), color="black", lw=2, label="Distribución t bajo H0")
        
        x_rech = np.linspace(t_crit_uni, 5, 200)
        ax.fill_between(x_rech, stats.t.pdf(x_rech, df=gl), color="red", alpha=0.35, label=f"Zona de rechazo (t > {t_crit_uni:.2f})")
        ax.axvline(t_obs, color="blue", linestyle="--", lw=2, label=f"t obs = {t_obs:.2f}")
        
        ax.set_title(r"Prueba de Hipótesis: $\mu > 10$")
        ax.set_xlabel("Valor t")
        ax.set_ylabel("Densidad")
        ax.legend()
        self.fig.tight_layout()

        concepto = "Contraste de Hipótesis para una Muestra"
        explicacion = "¿El rendimiento medio poblacional en matemáticas es significativamente mayor a 10 (condición de aprobado)? Evaluamos la evidencia aportada por nuestra muestra."
        formula = "H0: μ = 10 \nH1: μ > 10 (unilateral derecha)\nt_obs = (x̄ - 10) / (S / √n)"
        resultado = f"t observado = {t_obs:.4f}\nt crítico = {t_crit_uni:.4f}\np-valor = {p_unilateral:.6f}"
        
        if p_unilateral <= ALFA:
            interpretacion = f"Como el p-valor ({p_unilateral:.4f}) es menor al nivel de significancia α (0.05), SE RECHAZA la hipótesis nula (H0). Hay evidencia estadística sólida de que el rendimiento medio poblacional en la materia es aprobatorio."
        else:
            interpretacion = "NO se rechaza H0. No hay evidencia estadística suficiente en esta muestra para afirmar que la nota media supera el 10."

        self._actualizar_texto(concepto, explicacion, formula, resultado, interpretacion)

    def mostrar_u10(self):
        self.lbl_titulo.config(text="Unidad 10 — Regresión Lineal Simple")
        
        reg = self.df[["nota_segundo_semestre", "nota_final"]].dropna()
        x = reg["nota_segundo_semestre"].to_numpy(dtype=float)
        y = reg["nota_final"].to_numpy(dtype=float)
        n_obs = len(x)
        
        r, p_pearson = stats.pearsonr(x, y)
        ajuste = stats.linregress(x, y)
        b1, b0 = ajuste.slope, ajuste.intercept
        r2 = ajuste.rvalue ** 2

        ax = self.fig.add_subplot(111)
        ax.scatter(x, y, alpha=0.75, edgecolor="black", color="#3498DB", label="Estudiantes")
        
        x_linea = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_linea, b0 + b1 * x_linea, color="#E74C3C", lw=2, label=rf"$\hat{{Y}} = {b0:.2f} + {b1:.2f} X$")
        
        ax.set_title(r"Regresión: 2° Semestre ($X$) vs Nota Final ($Y$)")
        ax.set_xlabel("X = Nota Segundo Semestre (G2, 0-20)")
        ax.set_ylabel("Y = Nota Final (G3, 0-20)")
        ax.legend()
        self.fig.tight_layout()

        concepto = "Regresión Lineal Simple y Correlación"
        explicacion = (
            "¿Existe una relación lineal fuerte entre el desempeño "
            "del alumno en el segundo semestre (G2) y su calificación definitiva (G3)?\n\n"
            "• X = nota_segundo_semestre\n"
            "• Y = nota_final"
        )
        formula = "Modelo Matemático:\n Y = β₀ + β₁X + ε\n Ecuación estimada: Ŷ = a + bX"
        resultado = (
            f"Cantidad de observaciones (n) = {n_obs}\n"
            f"Coef. correlación (r) = {r:.3f}\n"
            f"Coef. determinación (R²) = {r2:.3f}\n"
            f"Pendiente (β₁) = {b1:.3f}\n"
            f"Intercepto (β₀) = {b0:.3f}\n\n"
            f"Ecuación: Ŷ = {b0:.3f} + {b1:.3f}X"
        )
        interpretacion = (
            f"Dirección e Intensidad: El coeficiente r = {r:.3f} indica una asociación lineal positiva "
            "extremadamente fuerte.\n\n"
            f"R²: El modelo explica el {r2*100:.1f}% de la variabilidad observada en la nota final.\n\n"
            f"Pendiente: Por cada punto que el alumno suma en el segundo semestre, su nota final "
            f"estimada se incrementa en {b1:.3f} puntos. Esto muestra una alta asociación debido a que "
            "el desempeño histórico define fuertemente el cierre del año escolar."
        )

        self._actualizar_texto(concepto, explicacion, formula, resultado, interpretacion)
        
    def guardar_grafico(self):
            # Crea la carpeta 'graficos' en tu repo si no existe
            carpeta_graficos = Path(__file__).resolve().parent / "graficos"
            carpeta_graficos.mkdir(exist_ok=True)
            
            # Asigna un nombre al archivo según la unidad en la que estés
            nombres = ["u6_histograma", "u7_estimacion", "u8_intervalos", "u9_hipotesis", "u10_regresion"]
            nombre_archivo = f"{nombres[self.indice_actual]}.png"
            ruta = carpeta_graficos / nombre_archivo
            
            # Guarda la figura actual con buena resolución
            self.fig.savefig(str(ruta), dpi=150, bbox_inches="tight")
            
            # Tira un aviso cuando termina
            messagebox.showinfo("¡Guardado!", f"El gráfico se guardó con éxito")
# ===========================================================================
# ARRANQUE DE LA APLICACIÓN
# ===========================================================================
if __name__ == "__main__":
    try:
        datos = DatosEstadisticos()
        app = PresentacionEstadistica(datos)
        print("Iniciando entorno gráfico... Presioná la cruz de la ventana para salir.")
        app.mainloop()
    except Exception as error:
        print(f"Error crítico al arrancar la app: {error}")