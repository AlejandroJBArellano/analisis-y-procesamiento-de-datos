"""
Visualización de Outliers con Boxplots
========================================
Análisis matemático y visual de los "bigotes" en boxplots.

La fórmula para los límites de los whiskers:
    - Límite inferior: Q1 - 1.5 × IQR
    - Límite superior: Q3 + 1.5 × IQR
    
Cualquier punto fuera de estos límites se considera un OUTLIER.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# 1. GENERAR DATASET
# ============================================================================

np.random.seed(42)

# Datos normales
datos_normales = np.random.normal(loc=100, scale=15, size=200)

# Agregar outliers específicos para visualización
outliers = np.array([45, 50, 155, 160, 165, 170])
datos_con_outliers = np.concatenate([datos_normales, outliers])

print("=" * 80)
print("ANÁLISIS DE OUTLIERS CON BOXPLOTS")
print("=" * 80)

# ============================================================================
# 2. CALCULAR ESTADÍSTICAS Y LÍMITES DE WHISKERS
# ============================================================================

Q1 = np.percentile(datos_con_outliers, 25)
Q3 = np.percentile(datos_con_outliers, 75)
IQR = Q3 - Q1

# Fórmulas para los límites
limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

media = np.mean(datos_con_outliers)
mediana = np.median(datos_con_outliers)

print(f"\n📊 ESTADÍSTICAS DESCRIPTIVAS:")
print(f"   Media:               {media:.2f}")
print(f"   Mediana:             {mediana:.2f}")
print(f"   Mínimo:              {np.min(datos_con_outliers):.2f}")
print(f"   Máximo:              {np.max(datos_con_outliers):.2f}")

print(f"\n📦 CUARTILES E IQR:")
print(f"   Q1 (Percentil 25%):  {Q1:.2f}")
print(f"   Q2 (Mediana 50%):    {mediana:.2f}")
print(f"   Q3 (Percentil 75%):  {Q3:.2f}")
print(f"   IQR (Q3 - Q1):       {IQR:.2f}")

print(f"\n📍 LÍMITES DE LOS WHISKERS (Bigotes):")
print(f"   Fórmula inferior: Q1 - 1.5 × IQR")
print(f"                   = {Q1:.2f} - 1.5 × {IQR:.2f}")
print(f"                   = {Q1:.2f} - {1.5 * IQR:.2f}")
print(f"                   = {limite_inferior:.2f}")

print(f"\n   Fórmula superior: Q3 + 1.5 × IQR")
print(f"                   = {Q3:.2f} + 1.5 × {IQR:.2f}")
print(f"                   = {Q3:.2f} + {1.5 * IQR:.2f}")
print(f"                   = {limite_superior:.2f}")

# Identificar outliers
outliers_identificados = datos_con_outliers[(datos_con_outliers < limite_inferior) | 
                                           (datos_con_outliers > limite_superior)]

print(f"\n🎯 OUTLIERS IDENTIFICADOS:")
print(f"   Total de outliers: {len(outliers_identificados)}")
print(f"   Valores: {sorted(outliers_identificados)}")

# Clasificar outliers
outliers_bajos = outliers_identificados[outliers_identificados < limite_inferior]
outliers_altos = outliers_identificados[outliers_identificados > limite_superior]

print(f"\n   • Outliers por debajo de {limite_inferior:.2f}: {len(outliers_bajos)}")
if len(outliers_bajos) > 0:
    print(f"     Valores: {sorted(outliers_bajos)}")

print(f"\n   • Outliers por encima de {limite_superior:.2f}: {len(outliers_altos)}")
if len(outliers_altos) > 0:
    print(f"     Valores: {sorted(outliers_altos)}")

# ============================================================================
# 3. CREAR VISUALIZACIONES
# ============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

fig.suptitle('Análisis de Outliers con Boxplots - Fórmula IQR',
             fontsize=18, fontweight='bold', y=0.995)

# ============================================================================
# GRÁFICO 1: Boxplot Simple
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

bp = ax1.boxplot(datos_con_outliers, vert=True, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2.5),
                  whiskerprops=dict(color='black', linewidth=1.5),
                  capprops=dict(color='black', linewidth=1.5),
                  flierprops=dict(marker='o', markerfacecolor='red', 
                                markersize=8, linestyle='none', alpha=0.8),
                  widths=0.5)

# Añadir líneas de referencia
ax1.axhline(y=Q1, color='blue', linestyle='--', linewidth=1, alpha=0.5, label='Q1')
ax1.axhline(y=Q3, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Q3')
ax1.axhline(y=mediana, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Mediana')
ax1.axhline(y=limite_inferior, color='orange', linestyle=':', linewidth=2, 
            label=f'Límite Inferior ({limite_inferior:.1f})')
ax1.axhline(y=limite_superior, color='purple', linestyle=':', linewidth=2,
            label=f'Límite Superior ({limite_superior:.1f})')

ax1.set_ylabel('Valores', fontsize=11, fontweight='bold')
ax1.set_title('Boxplot con Líneas de Referencia', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 2: Boxplot Horizontal (versión claridad)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

bp2 = ax2.boxplot(datos_con_outliers, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7),
                   medianprops=dict(color='darkred', linewidth=2.5),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red',
                                 markersize=8, linestyle='none', alpha=0.8),
                   widths=0.5)

# Anotaciones en el boxplot horizontal
# Obtener información del boxplot
whiskers_data = bp2['whiskers']
whisker_low = whiskers_data[0].get_xdata()[1]
whisker_high = whiskers_data[1].get_xdata()[1]

ax2.text(whisker_low, 1.25, f'Bigote Inferior\n{whisker_low:.1f}', 
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
ax2.text(whisker_high, 1.25, f'Bigote Superior\n{whisker_high:.1f}',
         ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax2.set_xlabel('Valores', fontsize=11, fontweight='bold')
ax2.set_title('Boxplot Horizontal - Identificación de Whiskers', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# ============================================================================
# GRÁFICO 3: Distribución vs Boxplot
# ============================================================================
ax3 = fig.add_subplot(gs[1, :])

# Histograma
ax3_twin = ax3.twinx()
ax3.hist(datos_con_outliers, bins=35, color='skyblue', edgecolor='black', alpha=0.6, label='Datos')

# Boxplot superpuesto (creado manualmente para visualización)
box_height = 40  # Altura del box en términos del histograma
box_y = np.max(ax3.get_ylim()) * 0.85

# Dibujar el boxplot manualmente
# Caja principal (IQR)
rect = plt.Rectangle((Q1, box_y - box_height/2), IQR, box_height,
                      fill=True, facecolor='lightblue', edgecolor='black', linewidth=2, alpha=0.7)
ax3.add_patch(rect)

# Línea de mediana
ax3.vlines(mediana, box_y - box_height/2, box_y + box_height/2, 
          colors='red', linewidth=3, label='Mediana')

# Whiskers
ax3.hlines(box_y, limite_inferior, limite_superior, 
          colors='black', linewidth=2, label='Range de Whiskers')
ax3.vlines(limite_inferior, box_y - box_height/4, box_y + box_height/4,
          colors='black', linewidth=2)
ax3.vlines(limite_superior, box_y - box_height/4, box_y + box_height/4,
          colors='black', linewidth=2)

# Outliers
ax3.scatter(outliers_identificados, [box_y]*len(outliers_identificados),
           color='red', s=150, marker='o', zorder=5, label='Outliers', edgecolors='darkred', linewidth=2)

# Líneas punteadas para los límites
ax3.axvline(limite_inferior, color='orange', linestyle=':', linewidth=2, alpha=0.7)
ax3.axvline(limite_superior, color='purple', linestyle=':', linewidth=2, alpha=0.7)

ax3.set_xlabel('Valores', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
ax3.set_title('Distribución de Datos + Boxplot Superpuesto', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3, axis='x')

# ============================================================================
# GRÁFICO 4: Fórmula Visual
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')

# Texto explicativo de la fórmula
formula_text = f"""
FÓRMULA DE LOS WHISKERS (BIGOTES)

Mínimo de los datos = {np.min(datos_con_outliers):.2f}
Máximo de los datos = {np.max(datos_con_outliers):.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cuartiles:
  Q1 (25%) = {Q1:.2f}  ← Inicio de la caja
  Q2 (50%) = {mediana:.2f}  ← Línea roja (mediana)
  Q3 (75%) = {Q3:.2f}  ← Final de la caja

IQR = Q3 - Q1 = {Q3:.2f} - {Q1:.2f} = {IQR:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LÍMITES DE WHISKERS:

Whisker Inferior:
  Q1 - 1.5 × IQR = {Q1:.2f} - 1.5 × {IQR:.2f}
                 = {Q1:.2f} - {1.5*IQR:.2f}
                 = {limite_inferior:.2f}

Whisker Superior:
  Q3 + 1.5 × IQR = {Q3:.2f} + 1.5 × {IQR:.2f}
                 = {Q3:.2f} + {1.5*IQR:.2f}
                 = {limite_superior:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REGLA:
  Si valor < {limite_inferior:.2f}  → OUTLIER
  Si valor > {limite_superior:.2f}  → OUTLIER
"""

ax4.text(0.05, 0.95, formula_text, transform=ax4.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================================
# GRÁFICO 5: Recta Numérica con Outliers
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

# Crear recta numérica
x_min = min(datos_con_outliers) - 10
x_max = max(datos_con_outliers) + 10

ax5.set_xlim(x_min, x_max)
ax5.set_ylim(0, 10)

# Recta base
ax5.plot([x_min, x_max], [5, 5], 'k-', linewidth=2)

# Marcar puntos importantes
puntos = {
    'Whisker Inf': (limite_inferior, 'orange'),
    'Q1': (Q1, 'blue'),
    'Mediana': (mediana, 'red'),
    'Q3': (Q3, 'green'),
    'Whisker Sup': (limite_superior, 'purple')
}

for i, (label, (valor, color)) in enumerate(puntos.items()):
    ax5.plot(valor, 5, 'o', markersize=12, color=color)
    altura = 7 if i % 2 == 0 else 3
    ax5.annotate(f'{label}\n{valor:.1f}', xy=(valor, 5), xytext=(valor, altura),
                ha='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.6),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Marcar región "no outliers"
ax5.axvspan(limite_inferior, limite_superior, alpha=0.2, color='green', label='Región sin outliers')

# Marcar outliers
for outlier in outliers_identificados:
    ax5.plot(outlier, 5, 'rx', markersize=15, markeredgewidth=3)

ax5.set_ylabel('')
ax5.set_xlabel('Valores', fontsize=11, fontweight='bold')
ax5.set_title('Recta Numérica: Visualización de Outliers', fontsize=12, fontweight='bold')
ax5.set_yticks([])
ax5.grid(True, alpha=0.3, axis='x')
ax5.legend(loc='upper right', fontsize=9)

# ============================================================================
# 4. GUARDAR VISUALIZACIÓN
# ============================================================================

plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/boxplot_outliers.png',
            dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado como: boxplot_outliers.png")
plt.close()

# ============================================================================
# 5. ANÁLISIS CON DATAFRAME
# ============================================================================

print(f"\n" + "=" * 80)
print("ANÁLISIS CON DATAFRAME")
print("=" * 80)

# Crear dataframe
df = pd.DataFrame({
    'valores': datos_con_outliers,
    'tipo': ['Normal'] * len(datos_normales) + ['Outlier'] * len(outliers)
})

print(f"\nResumen estadístico:")
print(df['valores'].describe().round(2))

print(f"\nDesglose por tipo:")
print(df.groupby('tipo')['valores'].describe().round(2))

# ============================================================================
# 6. USANDO SEABORN PARA BOXPLOT ADICIONAL
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot con seaborn
sns.boxplot(y=datos_con_outliers, ax=axes[0], palette='Set2')
axes[0].set_title('Boxplot con Seaborn', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Valores', fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Violinplot para comparación
sns.violinplot(y=datos_con_outliers, ax=axes[1], palette='Set1')
axes[1].set_title('Violinplot (Distribución Kernel Density)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Valores', fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/seaborn_plots.png',
            dpi=300, bbox_inches='tight')
print(f"✅ Gráfico seaborn guardado como: seaborn_plots.png")
plt.close()

print(f"\n" + "=" * 80)
