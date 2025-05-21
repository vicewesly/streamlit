import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración básica de la página
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Configuración simple para los gráficos
sns.set_style("whitegrid")

# #################################################
# # CONFIGURACIÓN DEL DASHBOARD
# #################################################

# # Configuración básica de la página
st.title('Dashboard de Análisis de Ventas - Tiendas de Conveniencia')
st.markdown("""
Este dashboard permite analizar los datos macroeconómicos y su impacto en las ventas de tiendas de conveniencia.
Utilice los filtros y opciones para explorar diferentes visualizaciones y análisis.
""")

# #################################################
# # CARGA DE DATOS
# #################################################

# # Función para cargar datos con cache para mejorar rendimiento
@st.cache_data
def cargar_datos():
    """
    Carga el archivo CSV con datos macroeconómicos
    """
    df = pd.read_csv("USMacroG_v2.csv")
    # Usamos solo el año como referencia temporal
    df["Fecha"] = df["Year"]
    
    # Convertimos el año a datetime para mejor manejo
    df['Fecha'] = pd.to_datetime(df['Fecha'], format='%Y')
    return df

# Ejecutamos la función de carga
try:
    df = cargar_datos()
    st.success('Datos cargados correctamente')
    
    # Mostramos un vistazo de los datos
    with st.expander("Vista previa de los datos"):
        st.dataframe(df.head())
        
        # Información sobre las columnas
        col_info = pd.DataFrame({
            'Columna': df.columns,
            'Tipo': df.dtypes,
            'Valores no nulos': df.count(),
            'Valores únicos': [df[col].nunique() for col in df.columns]
        })
        st.write("Información de columnas:")
        st.dataframe(col_info)
        
        # Estadísticas descriptivas
        st.write("Estadísticas descriptivas:")
        st.dataframe(df.describe())
        
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# #################################################
# # BARRA LATERAL - FILTROS Y OPCIONES
# #################################################

st.sidebar.header('Filtros y Opciones')

# Filtro por rango de años
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider(
    'Seleccione rango de años:',
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filtramos el dataframe por el rango de años seleccionado
filtered_df = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

# Selección de variables para análisis
numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_columns = [col for col in numeric_columns if col != 'Year']  # Excluimos el año de las variables numéricas

# Selección de variables para diferentes análisis
st.sidebar.subheader('Selección de Variables')
selected_variables = st.sidebar.multiselect(
    'Variables para análisis básico:',
    options=numeric_columns,
    default=numeric_columns[:4] if len(numeric_columns) >= 4 else numeric_columns
)

# Variables para visualización 3D
if len(numeric_columns) >= 3:
    st.sidebar.subheader('Visualización 3D')
    var_x = st.sidebar.selectbox('Variable X:', numeric_columns, index=0)
    var_y = st.sidebar.selectbox('Variable Y:', numeric_columns, index=1 if len(numeric_columns) > 1 else 0)
    var_z = st.sidebar.selectbox('Variable Z:', numeric_columns, index=2 if len(numeric_columns) > 2 else 0)

# #################################################
# # CONTENIDO PRINCIPAL - VISUALIZACIONES
# #################################################

# Organizamos el dashboard en pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visualización Básica", 
    "Gráficos Compuestos", 
    "Análisis Multivariado", 
    "Visualización 3D",
    "Insights y Conclusiones"
])

# Tab 1: Visualización Básica
with tab1:
    st.header('Visualización Básica de Datos')
    
    # Dividimos la pantalla en dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Tendencias Temporales')
        # Gráfico de líneas para tendencias temporales
        if selected_variables:
            fig, ax = plt.subplots(figsize=(10, 6))
            for var in selected_variables:
                ax.plot(filtered_df['Year'], filtered_df[var], marker='o', linewidth=2, label=var)
            
            ax.set_title('Tendencias Temporales de Variables Seleccionadas')
            ax.set_xlabel('Año')
            ax.set_ylabel('Valor')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)
        else:
            st.warning('Seleccione al menos una variable para visualizar tendencias temporales.')
    
    with col2:
        st.subheader('Distribución de Variables')
        # Box plots para distribución de variables
        if selected_variables:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=filtered_df[selected_variables], ax=ax)
            
            ax.set_title('Distribución de Variables Seleccionadas')
            ax.set_ylabel('Valor')
            plt.xticks(rotation=45)
            
            st.pyplot(fig)
        else:
            st.warning('Seleccione al menos una variable para visualizar distribuciones.')
    
    # Gráfico de dispersión
    st.subheader('Relaciones entre Variables')
    if len(selected_variables) >= 2:
        scatter_x = st.selectbox('Variable X para dispersión:', selected_variables, index=0)
        scatter_y = st.selectbox('Variable Y para dispersión:', selected_variables, index=1 if len(selected_variables) > 1 else 0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=scatter_x, y=scatter_y, data=filtered_df, ax=ax, 
                       hue='Year' if 'Year' in filtered_df.columns else None,
                       palette='viridis', s=100, alpha=0.7)
        
        ax.set_title(f'Relación entre {scatter_x} y {scatter_y}')
        ax.set_xlabel(scatter_x)
        ax.set_ylabel(scatter_y)
        ax.grid(True)
        
        st.pyplot(fig)
    else:
        st.warning('Seleccione al menos dos variables para visualizar relaciones.')

# Tab 2: Gráficos Compuestos
with tab2:
    st.header('Gráficos Compuestos y Contextualización')
    
    st.markdown("""
    En esta sección se presentan gráficos compuestos que permiten analizar múltiples variables y 
    relaciones simultáneamente, proporcionando un contexto más completo para la interpretación de los datos.
    """)
    
    # Gráfico combinado: línea temporal y barras
    if len(selected_variables) >= 2:
        st.subheader('Evolución Temporal Combinada')
        
        var1 = st.selectbox('Variable principal (línea):', selected_variables, index=0)
        var2 = st.selectbox('Variable secundaria (barras):', selected_variables, index=1 if len(selected_variables) > 1 else 0)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Gráfico de línea para la primera variable
        color = 'tab:blue'
        ax1.set_xlabel('Año')
        ax1.set_ylabel(var1, color=color)
        ax1.plot(filtered_df['Year'], filtered_df[var1], color=color, marker='o')
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Gráfico de barras para la segunda variable
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel(var2, color=color)
        ax2.bar(filtered_df['Year'], filtered_df[var2], color=color, alpha=0.3)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Evolución comparativa de {var1} y {var2}')
        
        st.pyplot(fig)
    
    # Matriz de correlación
    st.subheader('Matriz de Correlación')
    
    if selected_variables:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = filtered_df[selected_variables].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', linewidths=0.5)
        ax.set_title('Matriz de Correlación entre Variables Seleccionadas')
        
        st.pyplot(fig)
        
        # Explicación de la matriz de correlación
        st.markdown("""
        La matriz de correlación muestra la fuerza de la relación entre las variables seleccionadas:
        - Valores cercanos a 1 indican una fuerte correlación positiva
        - Valores cercanos a -1 indican una fuerte correlación negativa
        - Valores cercanos a 0 indican poca o ninguna correlación
        """)
    else:
        st.warning('Seleccione al menos dos variables para generar la matriz de correlación.')

# Tab 3: Análisis Multivariado
with tab3:
    st.header('Visualización de Datos Multivariados')
    
    st.markdown("""
    Esta sección utiliza técnicas avanzadas para visualizar relaciones complejas entre múltiples variables 
    y reducir la dimensionalidad de los datos para facilitar su interpretación.
    """)
    
    # Análisis de Componentes Principales (PCA)
    if len(selected_variables) >= 3:
        st.subheader('Análisis de Componentes Principales (PCA)')
        
        # Escalamos los datos para PCA
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(filtered_df[selected_variables])
        
        # Aplicamos PCA
        n_components = min(3, len(selected_variables))
        pca = PCA(n_components=n_components)
        componentes_principales = pca.fit_transform(df_scaled)
        
        # Creamos un dataframe con los componentes principales
        df_pca = pd.DataFrame(
            data=componentes_principales,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Añadimos el año para referencia
        df_pca['Year'] = filtered_df['Year'].values
        
        # Visualizamos los componentes principales
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = sns.scatterplot(
            x='PC1', 
            y='PC2', 
            data=df_pca, 
            hue='Year',
            palette='viridis', 
            s=100, 
            alpha=0.7,
            ax=ax
        )
        
        # Añadimos etiquetas y título
        ax.set_title('Análisis de Componentes Principales (PCA)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} de varianza)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} de varianza)')
        ax.grid(True)
        
        st.pyplot(fig)
        
        # Visualización de la importancia de cada variable en los componentes principales
        st.subheader('Contribución de Variables a Componentes Principales')
        loadings = pd.DataFrame(
            pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(n_components)], 
            index=selected_variables
        )
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(loadings, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title('Contribución de Variables a Componentes Principales')
        
        st.pyplot(fig)
        
        # Explicación del PCA
        st.markdown(f"""
        **Interpretación del Análisis de Componentes Principales:**
        
        El PCA ha reducido la dimensionalidad de los datos, manteniendo un {sum(pca.explained_variance_ratio_):.2%} de la varianza original:
        - El primer componente principal (PC1) explica un {pca.explained_variance_ratio_[0]:.2%} de la varianza
        - El segundo componente principal (PC2) explica un {pca.explained_variance_ratio_[1]:.2%} de la varianza
        {"- El tercer componente principal (PC3) explica un " + f"{pca.explained_variance_ratio_[2]:.2%} de la varianza" if n_components > 2 else ""}
        
        La matriz de contribuciones muestra cómo cada variable original influye en los componentes principales,
        lo que permite identificar qué variables tienen mayor impacto en la variabilidad de los datos.
        """)
    else:
        st.warning('Seleccione al menos tres variables para realizar el análisis de componentes principales.')

    # Análisis de clusters (opcional)
    if len(selected_variables) >= 2:
        st.subheader('Distribución Conjunta de Variables')
        
        # Seleccionar variables para el pairplot
        pair_vars = st.multiselect(
            'Seleccione variables para visualizar sus distribuciones conjuntas (máximo 4):',
            options=selected_variables,
            default=selected_variables[:min(3, len(selected_variables))]
        )
        
        if len(pair_vars) >= 2 and len(pair_vars) <= 4:
            # Pairplot para visualizar distribuciones conjuntas
            fig = sns.pairplot(filtered_df, vars=pair_vars, hue='Year' if 'Year' in filtered_df.columns else None, 
                              palette='viridis', diag_kind='kde')
            fig.fig.suptitle('Distribuciones Conjuntas de Variables Seleccionadas', y=1.02)
            
            st.pyplot(fig)
        elif len(pair_vars) > 4:
            st.warning('Por favor, seleccione máximo 4 variables para el análisis conjunto.')
        else:
            st.warning('Seleccione al menos 2 variables para visualizar distribuciones conjuntas.')

# Tab 4: Visualización 3D
with tab4:
    st.header('Visualización en 3D')
    
    st.markdown("""
    La visualización tridimensional permite observar relaciones complejas entre tres variables simultáneamente,
    revelando patrones que podrían no ser evidentes en representaciones bidimensionales.
    """)
    
    if len(numeric_columns) >= 3:
        # Gráfico de dispersión 3D
        st.subheader('Gráfico de Dispersión 3D')
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            filtered_df[var_x],
            filtered_df[var_y],
            filtered_df[var_z],
            c=filtered_df['Year'] if 'Year' in filtered_df.columns else None,
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.set_zlabel(var_z)
        ax.set_title(f'Visualización 3D: {var_x} vs {var_y} vs {var_z}')
        
        # Añadimos una barra de colores
        if 'Year' in filtered_df.columns:
            cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
            cbar.set_label('Año')
        
        # Mostramos el gráfico 3D
        st.pyplot(fig)
        
        # Justificación de la visualización 3D
        st.markdown(f"""
        **Justificación de la visualización 3D:**
        
        Se han seleccionado las variables {var_x}, {var_y} y {var_z} para la representación tridimensional
        porque permite analizar simultáneamente cómo estas tres variables interactúan entre sí.
        
        Beneficios de la visualización 3D:
        1. Revela patrones y relaciones que no son evidentes en visualizaciones 2D
        2. Permite identificar clusters o agrupaciones de datos en el espacio tridimensional
        3. Facilita la comprensión de interacciones complejas entre las tres variables
        4. Proporciona una perspectiva más completa sobre la estructura de los datos
        
        Esta visualización es especialmente útil para identificar tendencias no lineales y comprender
        cómo la tercera dimensión ({var_z}) afecta la relación entre {var_x} y {var_y}.
        """)
        
        # Opción para rotar la visualización 3D (conceptual, no implementable directamente en Streamlit)
        st.info("""
        Nota: Para una experiencia completa con la visualización 3D, ejecute este dashboard localmente,
        donde podrá rotar y manipular el gráfico 3D de forma interactiva.
        """)
    else:
        st.warning('No hay suficientes variables numéricas para realizar una visualización 3D.')

# Tab 5: Insights y Conclusiones
with tab5:
    st.header('Insights y Conclusiones')
    
    st.markdown("""
    En esta sección se presentan los principales hallazgos y conclusiones derivados del análisis
    de los datos, así como recomendaciones para la cadena de tiendas de conveniencia.
    """)
    
    # Principales hallazgos
    st.subheader('Principales Hallazgos')
    
    # Análisis de correlación para detectar relaciones importantes
    if len(selected_variables) >= 2:
        corr_matrix = filtered_df[selected_variables].corr()
        
        # Encontramos las correlaciones más fuertes (positivas y negativas)
        corr_unstack = corr_matrix.unstack()
        corr_unstack = corr_unstack[corr_unstack < 1.0]  # Eliminamos la diagonal (correlación = 1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Correlaciones más fuertes (positivas):**")
            st.dataframe(corr_unstack.nlargest(5))
        
        with col2:
            st.write("**Correlaciones más fuertes (negativas):**")
            st.dataframe(corr_unstack.nsmallest(5))
    
    # Hallazgos clave
    st.markdown("""
    **Hallazgos clave del análisis:**
    
    1. **Patrones temporales:** Se identificaron tendencias claras en las variables económicas a lo largo del tiempo.
    2. **Relaciones entre variables:** Existen correlaciones significativas entre ciertos indicadores económicos.
    3. **Componentes principales:** El análisis PCA reveló las variables que más contribuyen a la variabilidad de los datos.
    4. **Visualización 3D:** Permitió identificar relaciones complejas entre tres variables clave.
    """)
    
    # Recomendaciones
    st.subheader('Recomendaciones para la Cadena de Tiendas')
    
    st.markdown("""
    **Recomendaciones estratégicas basadas en el análisis:**
    
    1. **Planificación estacional:** Ajustar inventarios y promociones según los patrones temporales identificados.
    2. **Enfoque en variables clave:** Prestar especial atención a las variables con mayor impacto según el PCA.
    3. **Segmentación de mercado:** Utilizar los patrones identificados para adaptar estrategias a diferentes segmentos.
    4. **Monitoreo continuo:** Implementar un sistema de seguimiento de indicadores clave en tiempo real.
    5. **Estrategias personalizadas:** Desarrollar enfoques específicos basados en las correlaciones identificadas.
    """)
    
    # Limitaciones del análisis
    st.subheader('Limitaciones del Análisis')
    
    st.markdown("""
    **Algunas limitaciones a considerar:**
    
    1. **Datos históricos:** El análisis se basa en datos pasados, que pueden no reflejar condiciones futuras.
    2. **Variables no incluidas:** Podrían existir factores externos no considerados que afecten los resultados.
    3. **Correlación vs. causalidad:** Las correlaciones identificadas no necesariamente implican relaciones causales.
    4. **Granularidad temporal:** El análisis a nivel anual puede ocultar patrones estacionales más detallados.
    """)
    
    # Próximos pasos
    st.subheader('Próximos Pasos')
    
    st.markdown("""
    **Recomendaciones para análisis futuros:**
    
    1. **Incorporar más variables:** Incluir datos demográficos, meteorológicos y de competencia.
    2. **Análisis predictivo:** Desarrollar modelos para pronosticar ventas y comportamiento del consumidor.
    3. **Segmentación avanzada:** Aplicar técnicas de clustering para identificar patrones de comportamiento.
    4. **Dashboard en tiempo real:** Implementar un sistema de actualización automática de datos.
    5. **Análisis geoespacial:** Incorporar variables geográficas para identificar patrones regionales.
    """)

# Nota final en el footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
Dashboard desarrollado como parte de la tarea grupal de Visualización de Datos.
</div>
""", unsafe_allow_html=True)
