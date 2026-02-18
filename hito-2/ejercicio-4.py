"""
Pipeline de Transformación de Datos
====================================

Técnicas para "suavizar" y normalizar distribuciones con outliers extremos:
1. np.log1p() - Transformación logarítmica
2. np.sqrt() - Raíz cuadrada
3. Box-Cox - Transformación de potencia optimizada

Se demuestra cómo la "cola larga" de la distribución se comprime.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

# ============================================================================
# 1. GENERAR DATASET CON OUTLIERS EXTREMOS
# ============================================================================

print("=" * 80)
print("PIPELINE DE TRANSFORMACIÓN DE DATOS")
print("=" * 80)

np.random.seed(42)

# Generar una distribución sesgada (exponencial)
# que simula ingresos, tráfico web, etc.
n_base = 300
datos_base = np.random.exponential(scale=5000, size=n_base)

# Agregar algunos outliers extremos (multimillonarios, eventos raros, etc.)
outliers_extremos = np.array([
    100000, 150000, 200000, 300000, 500000, 750000, 1000000, 2000000
])

datos_originales = np.concatenate([datos_base, outliers_extremos])

print(f"\n📊 Dataset Original:")
print(f"   Muestras: {len(datos_originales)}")
print(f"   Mínimo: ${datos_originales.min():,.2f}")
print(f"   Máximo: ${datos_originales.max():,.2f}")
print(f"   Media: ${datos_originales.mean():,.2f}")
print(f"   Mediana: ${np.median(datos_originales):,.2f}")
print(f"   Desv. Est.: ${datos_originales.std():,.2f}")
print(f"   Rango: ${datos_originales.max() - datos_originales.min():,.2f}")

# Calcular statistics de sesgo
skewness_original = stats.skew(datos_originales)
kurtosis_original = stats.kurtosis(datos_originales)

print(f"\n📈 Estadísticas de Forma:")
print(f"   Skewness (sesgo): {skewness_original:.4f}")
print(f"      Interpretación: {'Muy sesgado a la derecha' if skewness_original > 1 else 'Sesgado' if skewness_original > 0.5 else 'Ligeramente sesgado'}")
print(f"   Kurtosis (curtosis): {kurtosis_original:.4f}")
print(f"      Interpretación: {'Colas muy pesadas' if kurtosis_original > 1 else 'Colas más pesadas'}")

# ============================================================================
# 2. APLICAR TRANSFORMACIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("APLICANDO TRANSFORMACIONES")
print("=" * 80)

# Transformación 1: log1p (log(1 + x))
datos_log1p = np.log1p(datos_originales)

print(f"\n1️⃣ TRANSFORMACIÓN LOG1P (log(1 + x)):")
print(f"   Mínimo: {datos_log1p.min():.4f}")
print(f"   Máximo: {datos_log1p.max():.4f}")
print(f"   Media: {datos_log1p.mean():.4f}")
print(f"   Mediana: {np.median(datos_log1p):.4f}")
print(f"   Desv. Est.: {datos_log1p.std():.4f}")

skewness_log1p = stats.skew(datos_log1p)
kurtosis_log1p = stats.kurtosis(datos_log1p)
print(f"   Skewness: {skewness_log1p:.4f} (antes: {skewness_original:.4f})")
print(f"   Kurtosis: {kurtosis_log1p:.4f} (antes: {kurtosis_original:.4f})")

# Transformación 2: sqrt (raíz cuadrada)
datos_sqrt = np.sqrt(datos_originales)

print(f"\n2️⃣ TRANSFORMACIÓN SQRT (√x):")
print(f"   Mínimo: {datos_sqrt.min():.4f}")
print(f"   Máximo: {datos_sqrt.max():.4f}")
print(f"   Media: {datos_sqrt.mean():.4f}")
print(f"   Mediana: {np.median(datos_sqrt):.4f}")
print(f"   Desv. Est.: {datos_sqrt.std():.4f}")

skewness_sqrt = stats.skew(datos_sqrt)
kurtosis_sqrt = stats.kurtosis(datos_sqrt)
print(f"   Skewness: {skewness_sqrt:.4f} (antes: {skewness_original:.4f})")
print(f"   Kurtosis: {kurtosis_sqrt:.4f} (antes: {kurtosis_original:.4f})")

# Transformación 3: Box-Cox (requiere valores > 0)
datos_boxcox, lambda_boxcox = stats.boxcox(datos_originales + 1)

print(f"\n3️⃣ TRANSFORMACIÓN BOX-COX:")
print(f"   Lambda óptimo: {lambda_boxcox:.4f}")
print(f"   Mínimo: {datos_boxcox.min():.4f}")
print(f"   Máximo: {datos_boxcox.max():.4f}")
print(f"   Media: {datos_boxcox.mean():.4f}")
print(f"   Mediana: {np.median(datos_boxcox):.4f}")
print(f"   Desv. Est.: {datos_boxcox.std():.4f}")

skewness_boxcox = stats.skew(datos_boxcox)
kurtosis_boxcox = stats.kurtosis(datos_boxcox)
print(f"   Skewness: {skewness_boxcox:.4f} (antes: {skewness_original:.4f})")
print(f"   Kurtosis: {kurtosis_boxcox:.4f} (antes: {kurtosis_original:.4f})")

# ============================================================================
# 3. CREAR DATAFRAME CON TODAS LAS TRANSFORMACIONES
# ============================================================================

df = pd.DataFrame({
    'original': datos_originales,
    'log1p': datos_log1p,
    'sqrt': datos_sqrt,
    'boxcox': datos_boxcox
})

# ============================================================================
# 4. VISUALIZACIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("GENERANDO VISUALIZACIONES...")
print("=" * 80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

fig.suptitle('Pipeline de Transformación: Compresión de Outliers Extremos',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# FILA 1: Histogramas
# ============================================================================

transformations = [
    ('original', datos_originales, 'Original', 'skyblue'),
    ('log1p', datos_log1p, 'Log1p: log(1 + x)', 'lightcoral'),
    ('sqrt', datos_sqrt, 'Sqrt: √x', 'lightgreen'),
    ('boxcox', datos_boxcox, f'Box-Cox (λ={lambda_boxcox:.3f})', 'lightyellow')
]

for idx, (key, data, title, color) in enumerate(transformations):
    ax = fig.add_subplot(gs[0, idx])
    
    ax.hist(data, bins=40, color=color, edgecolor='black', alpha=0.7)
    ax.set_title(f'{title}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Agregar estadísticas
    stats_text = f"μ: {np.mean(data):,.0f}\nσ: {np.std(data):,.0f}\nSkew: {stats.skew(data):.2f}"
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ============================================================================
# FILA 2: Density Plots (KDE)
# ============================================================================

for idx, (key, data, title, color) in enumerate(transformations):
    ax = fig.add_subplot(gs[1, idx])
    
    # KDE plot
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    density = kde(x_range)
    
    ax.fill_between(x_range, density, alpha=0.6, color=color, edgecolor='black', linewidth=2)
    ax.set_title(f'{title} - KDE', fontweight='bold', fontsize=11)
    ax.set_ylabel('Densidad', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Marcar media y mediana
    mean_val = np.mean(data)
    median_val = np.median(data)
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Mediana: {median_val:.2f}')
    ax.legend(fontsize=8, loc='upper right')

# ============================================================================
# FILA 3: Q-Q Plots (Comparar con distribución normal)
# ============================================================================

for idx, (key, data, title, color) in enumerate(transformations):
    ax = fig.add_subplot(gs[2, idx])
    
    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'{title} - Q-Q Plot', fontweight='bold', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Cambiar color de los puntos
    for line in ax.get_lines():
        line.set_color(color)
        line.set_alpha(0.7)

# ============================================================================
# GUARDAR FIGURA PRINCIPAL
# ============================================================================

plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/transformacion_datos.png',
            dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: transformacion_datos.png")
plt.close()

# ============================================================================
# 5. COMPARACIÓN DIRECTA: Antes vs Después LOG1P
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Efecto de log1p: Compresión de la Cola Derecha',
              fontsize=14, fontweight='bold')

# Plot 1: Superposición de histogramas
ax = axes[0, 0]
ax.hist(datos_originales, bins=40, color='red', alpha=0.5, label='Original', density=True)
ax2 = ax.twinx()
ax2.hist(datos_log1p, bins=40, color='blue', alpha=0.5, label='Log1p', density=True)
ax.set_xlabel('Original', fontweight='bold')
ax2.set_xlabel('Log1p', fontweight='bold')
ax.set_ylabel('Densidad (Original)', fontweight='bold', color='red')
ax2.set_ylabel('Densidad (Log1p)', fontweight='bold', color='blue')
ax.set_title('Distribuciones Superpuestas', fontweight='bold')

# Plot 2: KDE comparativo
ax = axes[0, 1]
datos_originales_sample = datos_originales[datos_originales < 50000]  # Limitar para visualizar
ax.hist(datos_originales_sample, bins=40, color='red', alpha=0.3, density=True, label='Original')
kde_orig = gaussian_kde(datos_originales_sample)
x_range = np.linspace(datos_originales_sample.min(), datos_originales_sample.max(), 200)
ax.plot(x_range, kde_orig(x_range), color='red', linewidth=2, label='KDE Original')
ax.set_xlabel('Valor', fontweight='bold')
ax.set_ylabel('Densidad', fontweight='bold')
ax.set_title('Original (sin valores extremos > 50k)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Scatter plot - Transformación visual
ax = axes[1, 0]
indices = np.arange(len(datos_originales))
ax.scatter(indices, datos_originales, alpha=0.5, color='red', s=20, label='Original')
ax.set_xlabel('Índice de Muestra', fontweight='bold')
ax.set_ylabel('Valor', fontweight='bold')
ax.set_title('Datos Originales con Outliers Extremos', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Scatter plot - Después de transformación
ax = axes[1, 1]
ax.scatter(indices, datos_log1p, alpha=0.5, color='blue', s=20, label='Log1p')
ax.set_xlabel('Índice de Muestra', fontweight='bold')
ax.set_ylabel('Valor (log escala)', fontweight='bold')
ax.set_title('Datos Transformados con log1p', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/log1p_comparison.png',
            dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: log1p_comparison.png")
plt.close()

# ============================================================================
# 6. TABLA DE IMPACTO DE TRANSFORMACIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("RESUMEN DE IMPACTO DE TRANSFORMACIONES")
print("=" * 80)

impact_summary = pd.DataFrame({
    'Métrica': ['Media', 'Mediana', 'Desv. Est.', 'Rango', 'Skewness', 'Kurtosis'],
    'Original': [
        f"${datos_originales.mean():,.2f}",
        f"${np.median(datos_originales):,.2f}",
        f"${datos_originales.std():,.2f}",
        f"${datos_originales.max() - datos_originales.min():,.2f}",
        f"{stats.skew(datos_originales):.4f}",
        f"{stats.kurtosis(datos_originales):.4f}"
    ],
    'Log1p': [
        f"{datos_log1p.mean():.4f}",
        f"{np.median(datos_log1p):.4f}",
        f"{datos_log1p.std():.4f}",
        f"{datos_log1p.max() - datos_log1p.min():.4f}",
        f"{stats.skew(datos_log1p):.4f}",
        f"{stats.kurtosis(datos_log1p):.4f}"
    ],
    'Sqrt': [
        f"{datos_sqrt.mean():.4f}",
        f"{np.median(datos_sqrt):.4f}",
        f"{datos_sqrt.std():.4f}",
        f"{datos_sqrt.max() - datos_sqrt.min():.4f}",
        f"{stats.skew(datos_sqrt):.4f}",
        f"{stats.kurtosis(datos_sqrt):.4f}"
    ],
    'Box-Cox': [
        f"{datos_boxcox.mean():.4f}",
        f"{np.median(datos_boxcox):.4f}",
        f"{datos_boxcox.std():.4f}",
        f"{datos_boxcox.max() - datos_boxcox.min():.4f}",
        f"{stats.skew(datos_boxcox):.4f}",
        f"{stats.kurtosis(datos_boxcox):.4f}"
    ]
})

print("\n" + impact_summary.to_string(index=False))

# ============================================================================
# 7. ANÁLISIS DE REDUCCIÓN
# ============================================================================

print(f"\n" + "=" * 80)
print("ANÁLISIS DE REDUCCIÓN DE VARIABILIDAD")
print("=" * 80)

reduction_skewness_log1p = (1 - abs(stats.skew(datos_log1p)) / abs(stats.skew(datos_originales))) * 100
reduction_kurtosis_log1p = (1 - abs(stats.kurtosis(datos_log1p)) / abs(stats.kurtosis(datos_originales))) * 100

reduction_skewness_sqrt = (1 - abs(stats.skew(datos_sqrt)) / abs(stats.skew(datos_originales))) * 100
reduction_kurtosis_sqrt = (1 - abs(stats.kurtosis(datos_sqrt)) / abs(stats.kurtosis(datos_originales))) * 100

reduction_skewness_boxcox = (1 - abs(stats.skew(datos_boxcox)) / abs(stats.skew(datos_originales))) * 100
reduction_kurtosis_boxcox = (1 - abs(stats.kurtosis(datos_boxcox)) / abs(stats.kurtosis(datos_originales))) * 100

print(f"\n✨ MEJORA EN SKEWNESS (Sesgo):")
print(f"   Log1p:   {reduction_skewness_log1p:.2f}% de reducción")
print(f"   Sqrt:    {reduction_skewness_sqrt:.2f}% de reducción")
print(f"   Box-Cox: {reduction_skewness_boxcox:.2f}% de reducción")

print(f"\n✨ MEJORA EN KURTOSIS (Curtosis):")
print(f"   Log1p:   {reduction_kurtosis_log1p:.2f}% de reducción")
print(f"   Sqrt:    {reduction_kurtosis_sqrt:.2f}% de reducción")
print(f"   Box-Cox: {reduction_kurtosis_boxcox:.2f}% de reducción")

# ============================================================================
# 8. COMPRESIÓN DE COLA
# ============================================================================

print(f"\n" + "=" * 80)
print("COMPRESIÓN DE OUTLIERS EXTREMOS (Cola Derecha)")
print("=" * 80)

# Encontrar outliers extremos (> percentil 95)
p95_original = np.percentile(datos_originales, 95)
extreme_outliers_idx = datos_originales > p95_original

print(f"\n📊 Outliers Extremos (> Percentil 95: {p95_original:,.2f}):")
print(f"   Cantidad: {extreme_outliers_idx.sum()}")
print(f"   Rango original: [{datos_originales[extreme_outliers_idx].min():,.2f}, {datos_originales[extreme_outliers_idx].max():,.2f}]")
print(f"   Rango log1p:    [{datos_log1p[extreme_outliers_idx].min():.4f}, {datos_log1p[extreme_outliers_idx].max():.4f}]")
print(f"   Compresión: {(1 - (datos_log1p[extreme_outliers_idx].max() - datos_log1p[extreme_outliers_idx].min()) / (datos_originales[extreme_outliers_idx].max() - datos_originales[extreme_outliers_idx].min())) * 100:.2f}%")

# ============================================================================
# 9. CONCLUSIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("CONCLUSIONES Y RECOMENDACIONES")
print("=" * 80)

print(f"""
🎯 TRANSFORMACIONES APLICADAS:

1. LOG1P (log(1 + x)):
   ✅ Comprime efectivamente la cola derecha
   ✅ Transforma relaciones exponenciales en lineales
   ✅ Interpetable: log(ingresos) → % de cambio
   ✅ Mejor para: datos de ingresos, precios, tráfico
   ⚠️  Solo funciona con valores ≥ 0

2. SQRT (√x):
   ✅ Compresión moderada (menos agresiva que log)
   ✅ Preserva mejor la escala original
   ⚠️  Menos efectiva con outliers extremos
   ✅ Mejor para: datos de conteos, áreas

3. BOX-COX:
   ✅ Encuentra la transformación óptima automáticamente
   ✅ Lambda = {lambda_boxcox:.4f}
   ✅ Maximiza normalidad de la distribución
   ⚠️  Solo funciona con valores > 0
   ⚠️  Más computacionalmente intensivo

📈 RESULTADOS:
   • Reducción de sesgo (log1p): {reduction_skewness_log1p:.2f}%
   • Reducción de curtosis (log1p): {reduction_kurtosis_log1p:.2f}%
   • Compresión de outliers extremos: Rango reducido de {datos_originales[extreme_outliers_idx].max() - datos_originales[extreme_outliers_idx].min():,.0f} a {datos_log1p[extreme_outliers_idx].max() - datos_log1p[extreme_outliers_idx].min():.2f}

💡 RECOMENDACIÓN:
   Para datos con outliers extremos y distribución sesgada:
   1. Primero intentar LOG1P (simple, interpretable)
   2. Si log1p no es suficiente, probar BOX-COX
   3. Usar SQRT para compresión suave
   4. SIEMPRE visualizar antes y después
""")

print(f"\n" + "=" * 80)
