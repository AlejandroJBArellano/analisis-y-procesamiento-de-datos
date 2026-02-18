"""
Desafío de la Media vs. Mediana
================================
Análisis del impacto de valores atípicos (outliers) en las medidas de tendencia central.
Este script demuestra cómo la media es sensible a outliers mientras la mediana es robusta.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# 1. CREAR DATASET SINTÉTICO DE INGRESOS NORMALES
# ============================================================================

np.random.seed(42)  # Para reproducibilidad

# Parámetros de distribución normal
ingreso_promedio = 3000  # Ingreso promedio en unidades monetarias
desviacion_estandar = 800  # Desviación estándar

# Generar 1,000 registros con distribución normal
ingresos_normales = np.random.normal(
    loc=ingreso_promedio,
    scale=desviacion_estandar,
    size=1000
)

# Asegurar que no haya ingresos negativos
ingresos_normales = np.abs(ingresos_normales)

print("=" * 70)
print("DATASET ORIGINAL (SIN OUTLIERS)")
print("=" * 70)
print(f"Total de registros: {len(ingresos_normales)}")
print(f"Mínimo ingreso: ${ingresos_normales.min():.2f}")
print(f"Máximo ingreso: ${ingresos_normales.max():.2f}")

# Calcular estadísticas ANTES de añadir outliers
media_antes = np.mean(ingresos_normales)
mediana_antes = np.median(ingresos_normales)

print(f"\n📊 ESTADÍSTICAS INICIALES:")
print(f"   Media:   ${media_antes:.2f}")
print(f"   Mediana: ${mediana_antes:.2f}")
print(f"   Diferencia: ${media_antes - mediana_antes:.2f}")

# ============================================================================
# 2. AÑADIR 5 MULTIMILLONARIOS (OUTLIERS)
# ============================================================================

# Los multimillonarios tienen ingresos 100 veces superiores al promedio
ingreso_multimillonario = ingreso_promedio * 100

ingresos_multimillonarios = np.array([ingreso_multimillonario] * 5)

print(f"\n" + "=" * 70)
print("AGREGANDO MULTIMILLONARIOS")
print("=" * 70)
print(f"Cantidad de multimillonarios: {len(ingresos_multimillonarios)}")
print(f"Ingreso de cada multimillonario: ${ingreso_multimillonario:.2f}")

# Combinar datos originales con outliers
ingresos_con_outliers = np.concatenate([ingresos_normales, ingresos_multimillonarios])

print(f"\nTotal de registros (después de añadir outliers): {len(ingresos_con_outliers)}")
print(f"Nuevo mínimo: ${ingresos_con_outliers.min():.2f}")
print(f"Nuevo máximo: ${ingresos_con_outliers.max():.2f}")

# Calcular estadísticas DESPUÉS de añadir outliers
media_despues = np.mean(ingresos_con_outliers)
mediana_despues = np.median(ingresos_con_outliers)

print(f"\n📊 ESTADÍSTICAS FINALES:")
print(f"   Media:   ${media_despues:.2f}")
print(f"   Mediana: ${mediana_despues:.2f}")
print(f"   Diferencia: ${media_despues - mediana_despues:.2f}")

# ============================================================================
# 3. ANÁLISIS COMPARATIVO
# ============================================================================

print(f"\n" + "=" * 70)
print("ANÁLISIS COMPARATIVO")
print("=" * 70)

cambio_media = ((media_despues - media_antes) / media_antes) * 100
cambio_mediana = ((mediana_despues - mediana_antes) / mediana_antes) * 100

print(f"\n📈 CAMBIO EN LA MEDIA:")
print(f"   Antes:  ${media_antes:.2f}")
print(f"   Después: ${media_despues:.2f}")
print(f"   Cambio: ${media_despues - media_antes:.2f} ({cambio_media:.2f}%)")

print(f"\n📈 CAMBIO EN LA MEDIANA:")
print(f"   Antes:  ${mediana_antes:.2f}")
print(f"   Después: ${mediana_despues:.2f}")
print(f"   Cambio: ${mediana_despues - mediana_antes:.2f} ({cambio_mediana:.2f}%)")

print(f"\n💡 CONCLUSIÓN:")
print(f"   La media se desplazó un {cambio_media:.2f}%")
print(f"   La mediana se desplazó solo un {cambio_mediana:.2f}%")
print(f"   Esto demuestra que la mediana es más ROBUSTA ante outliers")

# ============================================================================
# 4. CREAR VISUALIZACIONES
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Análisis de la Media vs. Mediana - Impacto de Outliers', 
             fontsize=16, fontweight='bold', y=0.995)

# Gráfico 1: Histograma ANTES de outliers
ax1 = axes[0, 0]
ax1.hist(ingresos_normales, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
ax1.axvline(media_antes, color='red', linestyle='--', linewidth=2.5, label=f'Media: ${media_antes:.2f}')
ax1.axvline(mediana_antes, color='green', linestyle='--', linewidth=2.5, label=f'Mediana: ${mediana_antes:.2f}')
ax1.set_title('ANTES: Dataset Original (Sin Outliers)', fontweight='bold', fontsize=12)
ax1.set_xlabel('Ingresos Mensuales ($)')
ax1.set_ylabel('Frecuencia')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Gráfico 2: Histograma DESPUÉS de outliers
ax2 = axes[0, 1]
ax2.hist(ingresos_con_outliers, bins=40, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(media_despues, color='red', linestyle='--', linewidth=2.5, label=f'Media: ${media_despues:.2f}')
ax2.axvline(mediana_despues, color='green', linestyle='--', linewidth=2.5, label=f'Mediana: ${mediana_despues:.2f}')
ax2.set_title('DESPUÉS: Con 5 Multimillonarios (Outliers)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Ingresos Mensuales ($)')
ax2.set_ylabel('Frecuencia')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# Gráfico 3: Comparación de Media y Mediana
ax3 = axes[1, 0]
categorias = ['Antes', 'Después']
medias = [media_antes, media_despues]
medianas = [mediana_antes, mediana_despues]

x = np.arange(len(categorias))
ancho = 0.35

barras1 = ax3.bar(x - ancho/2, medias, ancho, label='Media', color='red', alpha=0.7)
barras2 = ax3.bar(x + ancho/2, medianas, ancho, label='Mediana', color='green', alpha=0.7)

ax3.set_ylabel('Ingresos ($)', fontweight='bold')
ax3.set_title('Comparación: Media vs. Mediana', fontweight='bold', fontsize=12)
ax3.set_xticks(x)
ax3.set_xticklabels(categorias)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for barras in [barras1, barras2]:
    for barra in barras:
        altura = barra.get_height()
        ax3.text(barra.get_x() + barra.get_width()/2., altura,
                f'${altura:.0f}',
                ha='center', va='bottom', fontsize=9)

# Gráfico 4: Distribución con zoom en los datos normales
ax4 = axes[1, 1]
ax4.hist(ingresos_con_outliers, bins=60, color='mediumpurple', edgecolor='black', alpha=0.7)
ax4.axvline(media_despues, color='red', linestyle='--', linewidth=2.5, label=f'Media: ${media_despues:.2f}')
ax4.axvline(mediana_despues, color='green', linestyle='--', linewidth=2.5, label=f'Mediana: ${mediana_despues:.2f}')

# Limitar el eje X para ver mejor la distribución normal
ax4.set_xlim(0, 8000)
ax4.set_title('Distribución Completa (Vista Enfocada)', fontweight='bold', fontsize=12)
ax4.set_xlabel('Ingresos Mensuales ($)')
ax4.set_ylabel('Frecuencia')
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/media_vs_mediana.png', 
            dpi=300, bbox_inches='tight')
print("\n✅ Gráfico guardado como: media_vs_mediana.png")

plt.show()

# ============================================================================
# 5. CREAR DATAFRAME PARA ANÁLISIS ADICIONAL
# ============================================================================

# Crear dataframe con los datos
df = pd.DataFrame({'ingresos': ingresos_con_outliers})
df['tipo'] = ['Normal'] * 1000 + ['Multimillonario'] * 5

print(f"\n" + "=" * 70)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("=" * 70)
print("\nDataset Completo:")
print(df['ingresos'].describe())

print("\n\nDesglose por tipo:")
print(df.groupby('tipo')['ingresos'].describe().round(2))

# Calcular percentiles
print(f"\n📊 PERCENTILES:")
percentiles = [25, 50, 75, 90, 95, 99]
for p in percentiles:
    valor = np.percentile(ingresos_con_outliers, p)
    print(f"   Percentil {p:2d}: ${valor:10,.2f}")

# Calcular el rango intercuartílico (IQR)
q1 = np.percentile(ingresos_con_outliers, 25)
q3 = np.percentile(ingresos_con_outliers, 75)
iqr = q3 - q1

print(f"\n📦 RANGO INTERCUARTÍLICO (IQR):")
print(f"   Q1 (25%):  ${q1:.2f}")
print(f"   Q3 (75%):  ${q3:.2f}")
print(f"   IQR:       ${iqr:.2f}")

print(f"\n" + "=" * 70)
