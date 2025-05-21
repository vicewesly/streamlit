import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuración básica de la página
st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='Análisis de Ventas')

# Configuración simple para los gráficos
sns.set_style("whitegrid")
colors = px.colors.qualitative.Plotly

# #################################################
# # CONFIGURACIÓN DEL DASHBOARD
# #################################################

# Título y descripción
st.title('Dashboard de Análisis de Ventas')
st.markdown("""
Este dashboard permite analizar los datos de ventas de las tiendas de conveniencia, explorando patrones de ventas, 
preferencias de clientes y relaciones entre variables clave del negocio.
""")


# #################################################
# # CARGA DE DATOS
# #################################################

# Función para cargar datos con cache para mejorar rendimiento
@st.cache_data
def cargar_datos():
    """
    Carga el archivo CSV con datos de ventas
    """
    df = pd.read_csv("data.csv")

    # Convertimos la fecha a datetime para mejor manejo
    df['Date'] = pd.to_datetime(df['Date'])

    # Extraemos componentes de fecha
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.strftime('%B')
    df['Day'] = df['Date'].dt.day

    return df


# Ejecutamos la función de carga
try:
    df = cargar_datos()
    st.success('Datos cargados correctamente: {} registros'.format(len(df)))

    # Mostramos un vistazo de los datos
    with st.expander("Vista previa de los datos"):
        st.dataframe(df.head())

except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# #################################################
# # BARRA LATERAL - FILTROS Y OPCIONES
# #################################################

st.sidebar.header('Filtros')

# Filtro por mes
months = sorted(df['Month'].unique())
month_names = {m: datetime(2022, m, 1).strftime('%B') for m in months}
month_options = [month_names[m] for m in months]

selected_months = st.sidebar.multiselect(
    'Seleccione meses:',
    options=month_options,
    default=month_options
)

# Filtro por sucursal
all_branches = df['Branch'].unique().tolist()
selected_branches = st.sidebar.multiselect(
    'Seleccione sucursales:',
    options=all_branches,
    default=all_branches
)

# Filtro por línea de producto
all_product_lines = df['Product line'].unique().tolist()
selected_product_lines = st.sidebar.multiselect(
    'Seleccione líneas de producto:',
    options=all_product_lines,
    default=all_product_lines
)

# Filtro por tipo de cliente
all_customer_types = df['Customer type'].unique().tolist()
selected_customer_types = st.sidebar.multiselect(
    'Seleccione tipos de cliente:',
    options=all_customer_types,
    default=all_customer_types
)

# Filtro por método de pago
all_payment_methods = df['Payment'].unique().tolist()
selected_payment_methods = st.sidebar.multiselect(
    'Seleccione métodos de pago:',
    options=all_payment_methods,
    default=all_payment_methods
)

# Convertir nombres de meses de vuelta a números
selected_month_nums = []
if selected_months:
    selected_month_nums = [list(month_names.keys())[list(month_names.values()).index(m)] for m in selected_months]

# Aplicamos todos los filtros al dataframe
filtered_df = df.copy()

if selected_month_nums:
    filtered_df = filtered_df[filtered_df['Month'].isin(selected_month_nums)]
if selected_branches:
    filtered_df = filtered_df[filtered_df['Branch'].isin(selected_branches)]
if selected_product_lines:
    filtered_df = filtered_df[filtered_df['Product line'].isin(selected_product_lines)]
if selected_customer_types:
    filtered_df = filtered_df[filtered_df['Customer type'].isin(selected_customer_types)]
if selected_payment_methods:
    filtered_df = filtered_df[filtered_df['Payment'].isin(selected_payment_methods)]

# Verificamos si hay datos después de filtrar
if filtered_df.empty:
    st.warning("No hay datos que cumplan con los criterios de filtrado seleccionados. Por favor, ajuste los filtros.")
    st.stop()

# #################################################
# # KPIS GENERALES
# #################################################

st.header('KPIs Generales')

kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    st.metric(
        label="Total de Ventas",
        value=f"${filtered_df['Total'].sum():,.2f}"
    )

with kpi_col2:
    st.metric(
        label="Ingreso Bruto",
        value=f"${filtered_df['gross income'].sum():,.2f}"
    )

with kpi_col3:
    st.metric(
        label="Cantidad de Productos Vendidos",
        value=f"{filtered_df['Quantity'].sum():,}"
    )

with kpi_col4:
    st.metric(
        label="Calificación Promedio",
        value=f"{filtered_df['Rating'].mean():.2f}/10"
    )

# #################################################
# # ANÁLISIS REQUERIDOS
# #################################################

# 1. Evolución de las Ventas Totales
st.header('1. Evolución de las Ventas Totales')

# Agrupamos por fecha y calculamos suma diaria
daily_sales = filtered_df.groupby('Date')['Total'].sum().reset_index()

# Gráfico de evolución de ventas con Plotly
fig_daily_sales = px.line(
    daily_sales,
    x='Date',
    y='Total',
    title='Evolución de Ventas Totales por Día',
    labels={'Date': 'Fecha', 'Total': 'Ventas Totales ($)'},
    markers=True
)

fig_daily_sales.update_layout(
    xaxis_title='Fecha',
    yaxis_title='Ventas Totales ($)',
    hovermode='x unified',
    xaxis=dict(tickformat='%d %b %Y')
)

st.plotly_chart(fig_daily_sales, use_container_width=True)

st.markdown("""
Esta gráfica muestra la evolución de las ventas totales a lo largo del tiempo.
Permite identificar tendencias, picos y valles en el comportamiento de ventas.
""")

# 2. Ingresos por Línea de Producto
st.header('2. Ingresos por Línea de Producto')

# Agrupamos por línea de producto y calculamos suma
product_line_sales = filtered_df.groupby('Product line')['Total'].sum().sort_values(ascending=False).reset_index()

# Gráfico de barras para ingresos por línea de producto con Plotly
fig_product_line = px.bar(
    product_line_sales,
    x='Product line',
    y='Total',
    title='Ingresos Totales por Línea de Producto',
    labels={'Product line': 'Línea de Producto', 'Total': 'Ingresos Totales ($)'},
    color='Product line',
    text_auto='.2s'
)

fig_product_line.update_layout(
    xaxis_title='Línea de Producto',
    yaxis_title='Ingresos Totales ($)',
    xaxis={'categoryorder': 'total descending'}
)

st.plotly_chart(fig_product_line, use_container_width=True)

# Tabla con detalles adicionales por línea de producto
product_line_details = filtered_df.groupby('Product line').agg({
    'Total': 'sum',
    'Quantity': 'sum',
    'Invoice ID': 'count',
    'gross income': 'sum'
}).reset_index()

product_line_details = product_line_details.rename(columns={
    'Total': 'Ingresos Totales',
    'Quantity': 'Cantidad Vendida',
    'Invoice ID': 'Nº de Ventas',
    'gross income': 'Ingreso Bruto'
})

product_line_details = product_line_details.sort_values('Ingresos Totales', ascending=False)

st.dataframe(product_line_details, use_container_width=True)

st.markdown("""
Este análisis muestra los ingresos generados por cada línea de producto, permitiendo identificar 
las categorías más rentables para el negocio. La tabla proporciona detalles adicionales como 
cantidad vendida, número de ventas e ingreso bruto.
""")

# 3. Distribución de la Calificación de Clientes
st.header('3. Distribución de la Calificación de Clientes')

# Calculamos estadísticas descriptivas
rating_stats = filtered_df['Rating'].describe().to_frame().T
rating_stats = rating_stats.round(2)

st.dataframe(rating_stats, use_container_width=True)

col1, col2 = st.columns([3, 2])

with col1:
    # Histograma de calificaciones con Plotly
    fig_rating_hist = px.histogram(
        filtered_df,
        x='Rating',
        nbins=20,
        title='Distribución de Calificaciones de Clientes',
        labels={'Rating': 'Calificación', 'count': 'Frecuencia'},
        marginal='box',
        color_discrete_sequence=['#1E88E5']
    )

    fig_rating_hist.update_layout(
        xaxis_title='Calificación',
        yaxis_title='Frecuencia',
        bargap=0.1
    )

    st.plotly_chart(fig_rating_hist, use_container_width=True)

with col2:
    # Calificación promedio por línea de producto
    rating_by_product = filtered_df.groupby('Product line')['Rating'].mean().sort_values(ascending=False).reset_index()

    fig_rating_product = px.bar(
        rating_by_product,
        x='Rating',
        y='Product line',
        title='Calificación Promedio por Línea de Producto',
        labels={'Rating': 'Calificación Promedio', 'Product line': 'Línea de Producto'},
        orientation='h',
        color='Rating',
        color_continuous_scale='Blues'
    )

    fig_rating_product.update_layout(
        xaxis_title='Calificación Promedio',
        yaxis_title='Línea de Producto',
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig_rating_product, use_container_width=True)

st.markdown("""
Este análisis muestra la distribución de las calificaciones de los clientes. El histograma permite visualizar 
la frecuencia de cada puntuación, mientras que el gráfico de barras muestra la calificación promedio 
por línea de producto, ayudando a identificar qué categorías generan mayor satisfacción.
""")

# 4. Comparación del Gasto por Tipo de Cliente
st.header('4. Comparación del Gasto por Tipo de Cliente')

col1, col2 = st.columns(2)

with col1:
    # Boxplot para comparar distribución de gasto por tipo de cliente
    fig_customer_box = px.box(
        filtered_df,
        x='Customer type',
        y='Total',
        color='Customer type',
        title='Distribución del Gasto por Tipo de Cliente',
        labels={'Customer type': 'Tipo de Cliente', 'Total': 'Gasto Total ($)'}
    )

    fig_customer_box.update_layout(
        xaxis_title='Tipo de Cliente',
        yaxis_title='Gasto Total ($)'
    )

    st.plotly_chart(fig_customer_box, use_container_width=True)

with col2:
    # Estadísticas por tipo de cliente
    customer_stats = filtered_df.groupby('Customer type').agg({
        'Total': ['mean', 'median', 'std', 'sum'],
        'Invoice ID': 'count'
    }).reset_index()

    customer_stats.columns = ['Tipo de Cliente', 'Gasto Promedio', 'Gasto Mediano', 'Desviación Estándar',
                              'Gasto Total', 'Número de Compras']
    customer_stats = customer_stats.round(2)

    st.dataframe(customer_stats, use_container_width=True)

    # Gráfico de pastel para proporción de ventas por tipo de cliente
    fig_customer_pie = px.pie(
        customer_stats,
        values='Gasto Total',
        names='Tipo de Cliente',
        title='Proporción de Ventas por Tipo de Cliente',
        hole=0.4
    )

    st.plotly_chart(fig_customer_pie, use_container_width=True)

st.markdown("""
Este análisis compara el comportamiento de gasto entre clientes miembros y clientes normales.
El boxplot muestra la distribución del gasto, mientras que la tabla proporciona estadísticas detalladas
como promedio, mediana y desviación estándar. El gráfico circular muestra la proporción del gasto total
por tipo de cliente.
""")

# 5. Relación entre Costo y Ganancia Bruta
st.header('5. Relación entre Costo y Ganancia Bruta')

# Gráfico de dispersión con Plotly
fig_cogs_income = px.scatter(
    filtered_df,
    x='cogs',
    y='gross income',
    color='Branch',
    title='Relación entre Costo de Bienes Vendidos y Ganancia Bruta',
    labels={'cogs': 'Costo de Bienes Vendidos ($)', 'gross income': 'Ganancia Bruta ($)'},
    size='Total',
    hover_data=['Product line', 'Total', 'gross margin percentage']
)

fig_cogs_income.update_layout(
    xaxis_title='Costo de Bienes Vendidos ($)',
    yaxis_title='Ganancia Bruta ($)'
)

# Añadimos línea de tendencia
fig_cogs_income.update_traces(marker=dict(size=8))

st.plotly_chart(fig_cogs_income, use_container_width=True)

# Estadísticas de relación
correlation = filtered_df['cogs'].corr(filtered_df['gross income'])

st.info(f"Correlación entre el costo de bienes vendidos y la ganancia bruta: {correlation:.4f}")

st.markdown("""
Este gráfico de dispersión muestra la relación entre el costo de los bienes vendidos (COGS) y la ganancia bruta.
Cada punto representa una transacción, con el tamaño indicando el monto total de la venta.
La correlación positiva indica que generalmente las ventas con mayor costo generan mayor ganancia bruta.
""")

# 6. Métodos de Pago Preferidos
st.header('6. Métodos de Pago Preferidos')

col1, col2 = st.columns(2)

with col1:
    # Conteo de transacciones por método de pago
    payment_counts = filtered_df['Payment'].value_counts().reset_index()
    payment_counts.columns = ['Método de Pago', 'Frecuencia']

    fig_payment_counts = px.bar(
        payment_counts,
        x='Método de Pago',
        y='Frecuencia',
        title='Frecuencia de Uso de Métodos de Pago',
        color='Método de Pago',
        text_auto=True
    )

    fig_payment_counts.update_layout(
        xaxis_title='Método de Pago',
        yaxis_title='Número de Transacciones'
    )

    st.plotly_chart(fig_payment_counts, use_container_width=True)

with col2:
    # Monto total por método de pago
    payment_amount = filtered_df.groupby('Payment')['Total'].sum().reset_index()
    payment_amount.columns = ['Método de Pago', 'Monto Total']

    fig_payment_amount = px.pie(
        payment_amount,
        values='Monto Total',
        names='Método de Pago',
        title='Proporción del Monto Total por Método de Pago',
        hole=0.4
    )

    st.plotly_chart(fig_payment_amount, use_container_width=True)

# Estadísticas detalladas por método de pago
payment_stats = filtered_df.groupby('Payment').agg({
    'Total': ['mean', 'sum', 'count'],
    'Quantity': 'sum'
}).reset_index()

payment_stats.columns = ['Método de Pago', 'Compra Promedio', 'Monto Total', 'Número de Transacciones',
                         'Cantidad de Productos']
payment_stats = payment_stats.sort_values('Monto Total', ascending=False)
payment_stats = payment_stats.round(2)

st.dataframe(payment_stats, use_container_width=True)

st.markdown("""
Este análisis muestra las preferencias de los clientes en cuanto a métodos de pago.
El gráfico de barras indica la frecuencia de uso de cada método, mientras que el gráfico circular
muestra la proporción del monto total de ventas por método. La tabla proporciona estadísticas adicionales
como el valor promedio de compra y la cantidad total de productos vendidos con cada método.
""")


# 7. Análisis de Correlación Numérica
st.header('7. Análisis de Correlación Numérica')

# Nota aclaratoria sobre correlaciones perfectas
st.markdown("""
⚠️ **Nota:** Algunas correlaciones cercanas a 1.00 se deben a relaciones matemáticas directas entre variables.  
Por ejemplo, el `Total` incluye el `Tax 5%`, y el `gross income` se calcula a partir del `cogs`.  
Estas no representan una relación de comportamiento, sino una fórmula interna del negocio.
""")

# Seleccionamos las variables numéricas relevantes y filtramos derivadas para análisis más útil
filtered_corr_df = filtered_df[['Unit price', 'Quantity', 'Rating', 'Total']].corr()

# Heatmap de correlación con Plotly
fig_corr = px.imshow(
    filtered_corr_df,
    text_auto='.2f',
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title='Matriz de Correlación entre Variables Numéricas (sin variables derivadas)'
)

fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# Mostrar las correlaciones más altas (ignorando diagonales)
corr_pairs = filtered_corr_df.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]
high_corr = corr_pairs.abs().sort_values(ascending=False)[:10]

st.subheader("Pares con mayor correlación significativa:")
for idx, corr_value in high_corr.items():
    st.write(f"- **{idx[0]}** y **{idx[1]}**: {corr_value:.4f}")

st.header('7. Análisis de Correlación Numérica')

# Seleccionamos las variables numéricas relevantes
numeric_vars = ['Unit price', 'Quantity', 'Tax 5%', 'Total', 'cogs', 'gross income', 'Rating',
                'gross margin percentage']
corr_df = filtered_df[numeric_vars].corr()

# Heatmap de correlación con Plotly
fig_corr = px.imshow(
    corr_df,
    text_auto='.2f',
    aspect="auto",
    color_continuous_scale='RdBu_r',
    title='Matriz de Correlación entre Variables Numéricas'
)

fig_corr.update_layout(height=500)

st.plotly_chart(fig_corr, use_container_width=True)

# Pares de variables más correlacionadas
corr_pairs = corr_df.unstack()
corr_pairs = corr_pairs[corr_pairs < 1.0]  # Eliminamos diagonal
high_corr = corr_pairs.abs().sort_values(ascending=False)[:10]

st.subheader("Pares con mayor correlación:")
for idx, corr_value in high_corr.items():
    st.write(f"- **{idx[0]}** y **{idx[1]}**: {corr_value:.4f}")

st.markdown("""
La matriz de correlación muestra la fuerza y dirección de las relaciones lineales entre las variables numéricas.
Los valores cercanos a 1 indican una fuerte correlación positiva, valores cercanos a -1 indican una fuerte
correlación negativa, y valores cercanos a 0 indican poca o ninguna correlación lineal.

Este análisis es útil para identificar qué variables tienden a moverse juntas y en qué dirección,
lo que puede informar estrategias de negocio y decisiones operativas.
""")

# 8. Composición del Ingreso Bruto por Sucursal y Línea de Producto
st.header('8. Composición del Ingreso Bruto por Sucursal y Línea de Producto')

# Agrupamos datos por sucursal y línea de producto
branch_product_income = filtered_df.groupby(['Branch', 'Product line'])['gross income'].sum().reset_index()

# Gráfico de barras apiladas con Plotly
fig_branch_product = px.bar(
    branch_product_income,
    x='Branch',
    y='gross income',
    color='Product line',
    title='Composición del Ingreso Bruto por Sucursal y Línea de Producto',
    labels={'Branch': 'Sucursal', 'gross income': 'Ingreso Bruto ($)', 'Product line': 'Línea de Producto'},
    barmode='stack'
)

fig_branch_product.update_layout(
    xaxis_title='Sucursal',
    yaxis_title='Ingreso Bruto ($)',
    legend_title='Línea de Producto'
)

st.plotly_chart(fig_branch_product, use_container_width=True)

# Tabla detallada
branch_details = filtered_df.pivot_table(
    index='Branch',
    columns='Product line',
    values='gross income',
    aggfunc='sum',
    fill_value=0
).reset_index()

branch_total = filtered_df.groupby('Branch')['gross income'].sum().reset_index()
branch_total.columns = ['Branch', 'Total']

st.dataframe(branch_details, use_container_width=True)

# Análisis de proporción
branch_product_prop = filtered_df.groupby(['Branch', 'Product line'])['gross income'].sum().reset_index()
branch_totals = filtered_df.groupby('Branch')['gross income'].sum().reset_index()
branch_product_prop = branch_product_prop.merge(branch_totals, on='Branch', suffixes=('_product', '_total'))
branch_product_prop['percentage'] = (branch_product_prop['gross income_product'] / branch_product_prop[
    'gross income_total']) * 100

# Gráfico de proporciones
fig_prop = px.bar(
    branch_product_prop,
    x='Branch',
    y='percentage',
    color='Product line',
    title='Proporción de Ingreso Bruto por Línea de Producto en cada Sucursal',
    labels={'Branch': 'Sucursal', 'percentage': 'Porcentaje (%)', 'Product line': 'Línea de Producto'},
    barmode='stack'
)

fig_prop.update_layout(
    xaxis_title='Sucursal',
    yaxis_title='Porcentaje (%)',
    legend_title='Línea de Producto',
    yaxis=dict(ticksuffix='%')
)

st.plotly_chart(fig_prop, use_container_width=True)

st.markdown("""
Este análisis muestra cómo se compone el ingreso bruto en cada sucursal según las diferentes líneas de producto.
El primer gráfico muestra los valores absolutos, permitiendo comparar el desempeño total de cada sucursal.
El segundo gráfico muestra la proporción relativa de cada línea de producto dentro de cada sucursal,
lo que ayuda a identificar el enfoque o la especialización de cada ubicación.

Los datos en la tabla permiten un análisis más detallado de los valores específicos de ingreso bruto
por cada combinación de sucursal y línea de producto.
""")

# #################################################
# # CONCLUSIONES
# #################################################

st.header('Conclusiones y Recomendaciones')

st.markdown("""
### Hallazgos Clave:

1. **Ventas por Línea de Producto**: Se identificaron las líneas de producto más rentables, lo que permite optimizar la asignación de recursos y espacio en tienda.

2. **Patrones de Calificación**: La distribución de calificaciones de clientes indica el nivel general de satisfacción y áreas potenciales de mejora.

3. **Diferencias por Tipo de Cliente**: Los datos revelan diferencias significativas en el comportamiento de compra entre clientes miembros y clientes normales.

4. **Métodos de Pago**: Se observan preferencias claras en los métodos de pago utilizados, lo que puede informar estrategias de promoción y optimización de procesos.

5. **Correlaciones Significativas**: Las relaciones identificadas entre variables clave como costo y ganancia proporcionan insights para la optimización de precios y margen.

6. **Especialización por Sucursal**: Cada sucursal muestra un patrón único en la composición de sus ventas por línea de producto, lo que sugiere oportunidades para estrategias localizadas.

### Recomendaciones:

1. **Optimización de Inventario**: Ajustar la asignación de recursos según las líneas de producto más rentables.

2. **Programa de Fidelización**: Reforzar y expandir el programa de membresía, dadas las diferencias positivas en el comportamiento de compra de los miembros.

3. **Estrategias por Sucursal**: Desarrollar enfoques personalizados para cada sucursal basados en su composición única de ventas por línea de producto.

4. **Mejora de Experiencia**: Enfocarse en las líneas de producto con calificaciones más bajas para mejorar la satisfacción general del cliente.

5. **Promociones Específicas**: Crear ofertas especiales para métodos de pago menos utilizados pero con alto potencial.
""")

# Nota final en el footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.8em;">
Dashboard desarrollado para análisis de ventas de tiendas de conveniencia - 2025
</div>
""", unsafe_allow_html=True)