"""
Comparativa de Métodos de Detección de Outliers
================================================

Comparación entre:
1. Método IQR (Univariado) - Basado en cuartiles individuales
2. Isolation Forest (Multivariado) - Basado en el aislamiento de puntos anómalos

Dataset: Boston Housing Prices
(Precios de casas en la región de Boston)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# 1. CARGAR DATASET
# ============================================================================

print("=" * 80)
print("COMPARATIVA: IQR vs ISOLATION FOREST")
print("=" * 80)

# Usar Iris como dataset público (tiene características multivariadas)
iris = load_iris()
X = iris.data
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)

print(f"\n📊 Dataset: Iris Flowers")
print(f"   Dimensiones: {X.shape[0]} muestras × {X.shape[1]} características")
print(f"\n   Características:")
for i, name in enumerate(feature_names):
    print(f"   {i+1}. {name}")

print(f"\nResumen estadístico:")
print(df.describe().round(2))

# ============================================================================
# 2. MÉTODO 1: DETECCIÓN DE OUTLIERS CON IQR (Univariado)
# ============================================================================

print(f"\n" + "=" * 80)
print("MÉTODO 1: IQR (UNIVARIADO)")
print("=" * 80)

# Detectar outliers por cada característica
outliers_iqr_por_variable = {}
outliers_iqr_indices = set()

for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    
    # Encontrar outliers
    outliers_col = df[(df[col] < limite_inf) | (df[col] > limite_sup)].index.tolist()
    outliers_iqr_por_variable[col] = {
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'limite_inf': limite_inf,
        'limite_sup': limite_sup,
        'cantidad': len(outliers_col),
        'indices': outliers_col
    }
    
    outliers_iqr_indices.update(outliers_col)
    
    print(f"\n📌 {col}:")
    print(f"   Q1: {Q1:.3f}, Q3: {Q3:.3f}, IQR: {IQR:.3f}")
    print(f"   Límites: [{limite_inf:.3f}, {limite_sup:.3f}]")
    print(f"   Outliers detectados: {len(outliers_col)}")

outliers_iqr_unicos = len(outliers_iqr_indices)
print(f"\n🎯 TOTAL DE OUTLIERS IQR (UNIÓN): {outliers_iqr_unicos}")
print(f"   Índices: {sorted(outliers_iqr_indices)}")

# Crear columna de outliers IQR
df['outlier_iqr'] = df.index.isin(outliers_iqr_indices)

# ============================================================================
# 3. MÉTODO 2: DETECCIÓN CON ISOLATION FOREST (Multivariado)
# ============================================================================

print(f"\n" + "=" * 80)
print("MÉTODO 2: ISOLATION FOREST (MULTIVARIADO)")
print("=" * 80)

# Normalizar datos para Isolation Forest
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicar Isolation Forest
# contamination = "auto" detecta automáticamente porcentaje de outliers
iso_forest = IsolationForest(
    contamination=0.1,  # Asumir ~10% de outliers
    random_state=42,
    n_estimators=100
)

# -1 indica outlier, 1 indica normal
outlier_labels = iso_forest.fit_predict(X_scaled)
anomaly_scores = iso_forest.score_samples(X_scaled)

outliers_if_indices = set(np.where(outlier_labels == -1)[0].tolist())
outliers_if_unicos = len(outliers_if_indices)

print(f"\n📊 Isolation Forest Results:")
print(f"   Total de outliers detectados: {outliers_if_unicos}")
print(f"   Índices: {sorted(outliers_if_indices)}")
print(f"\n   Anomaly Scores (primeros 10):")
for i in range(min(10, len(anomaly_scores))):
    print(f"      Muestra {i}: {anomaly_scores[i]:.4f} {'[OUTLIER]' if i in outliers_if_indices else ''}")

# Crear columna de outliers Isolation Forest
df['outlier_if'] = df.index.isin(outliers_if_indices)
df['anomaly_score'] = anomaly_scores

# ============================================================================
# 4. COMPARACIÓN DE MÉTODOS
# ============================================================================

print(f"\n" + "=" * 80)
print("COMPARACIÓN DE MÉTODOS")
print("=" * 80)

# Outliers encontrados por ambos métodos
outliers_ambos = outliers_iqr_indices & outliers_if_indices
outliers_solo_iqr = outliers_iqr_indices - outliers_if_indices
outliers_solo_if = outliers_if_indices - outliers_iqr_indices

print(f"\n📊 ESTADÍSTICAS DE OVERLAPPING:")
print(f"   Outliers por IQR:           {outliers_iqr_unicos}")
print(f"   Outliers por IF:            {outliers_if_unicos}")
print(f"   Encontrados por AMBOS:      {len(outliers_ambos)}")
print(f"   Solo por IQR:               {len(outliers_solo_iqr)}")
print(f"   Solo por IF:                {len(outliers_solo_if)}")

print(f"\n🔗 ÍNDICES ÚNICOS DE IF (no detectados por IQR):")
if outliers_solo_if:
    print(f"   {sorted(outliers_solo_if)}")
else:
    print(f"   Ninguno - IF detectó los mismos outliers que IQR")

print(f"\n🔗 ÍNDICES ÚNICOS DE IQR (no detectados por IF):")
if outliers_solo_iqr:
    print(f"   {sorted(outliers_solo_iqr)}")
else:
    print(f"   Ninguno - IQR detectó los mismos outliers que IF")

# ============================================================================
# 5. ANÁLISIS DETALLADO DE ANOMALÍAS
# ============================================================================

print(f"\n" + "=" * 80)
print("ANÁLISIS DE ANOMALÍAS DETECTADAS")
print("=" * 80)

# Anomalías encontradas solo por Isolation Forest
if outliers_solo_if:
    print(f"\n🔴 Anomalías encontradas SOLO por Isolation Forest ({len(outliers_solo_if)}):")
    for idx in sorted(outliers_solo_if):
        print(f"\n   Índice {idx}:")
        for feat, val in zip(feature_names, X[idx]):
            print(f"      {feat}: {val:.3f}")
        print(f"      Anomaly Score: {anomaly_scores[idx]:.4f}")

# Anomalías encontradas solo por IQR
if outliers_solo_iqr:
    print(f"\n🔴 Anomalías encontradas SOLO por IQR ({len(outliers_solo_iqr)}):")
    for idx in sorted(outliers_solo_iqr):
        print(f"\n   Índice {idx}:")
        for feat, val in zip(feature_names, X[idx]):
            print(f"      {feat}: {val:.3f}")

# ============================================================================
# 6. VISUALIZACIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("GENERANDO VISUALIZACIONES...")
print("=" * 80)

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

fig.suptitle('Comparativa: IQR vs Isolation Forest - Detección de Outliers',
             fontsize=16, fontweight='bold', y=0.995)

# ============================================================================
# GRÁFICO 1-4: Comparación por variable (primeras 4 características)
# ============================================================================

for i, col in enumerate(feature_names[:4]):
    ax = fig.add_subplot(gs[0 if i < 2 else 1, i % 2])
    
    # Datos normales
    normal_data = df[~df['outlier_iqr'] & ~df['outlier_if']][col]
    
    # Outliers solo IQR
    outliers_iqr_only = df[df['outlier_iqr'] & ~df['outlier_if']][col]
    
    # Outliers solo IF
    outliers_if_only = df[~df['outlier_iqr'] & df['outlier_if']][col]
    
    # Outliers ambos
    outliers_both = df[df['outlier_iqr'] & df['outlier_if']][col]
    
    # Histograma
    ax.hist(normal_data, bins=20, color='skyblue', alpha=0.6, label='Normal', edgecolor='black')
    
    # Scatter dos outliers
    ax.scatter(outliers_both.values, [0]*len(outliers_both), 
              color='darkred', s=200, marker='*', label='Ambos', zorder=5, edgecolors='black', linewidth=2)
    ax.scatter(outliers_iqr_only.values, [-2]*len(outliers_iqr_only),
              color='orange', s=150, marker='s', label='Solo IQR', zorder=5, edgecolors='black', linewidth=1.5)
    ax.scatter(outliers_if_only.values, [2]*len(outliers_if_only),
              color='purple', s=150, marker='^', label='Solo IF', zorder=5, edgecolors='black', linewidth=1.5)
    
    # Líneas de límites IQR
    limits_info = outliers_iqr_por_variable[col]
    ax.axvline(limits_info['limite_inf'], color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Límites IQR')
    ax.axvline(limits_info['limite_sup'], color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax.set_title(f'{col}', fontweight='bold', fontsize=11)
    ax.set_ylabel('Frecuencia', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 5: Diagrama de Venn (Overlapping)
# ============================================================================
ax5 = fig.add_subplot(gs[0, 2])
ax5.axis('off')

# Texto de comparación
comparison_text = f"""
RESUMEN COMPARATIVO

IQR (Univariado):
  • Detecta: {outliers_iqr_unicos}
  
Isolation Forest (Multivariado):
  • Detecta: {outliers_if_unicos}
  
Overlapping:
  • Ambos métodos: {len(outliers_ambos)}
  • Solo IQR: {len(outliers_solo_iqr)}
  • Solo IF: {len(outliers_solo_if)}

Cobertura:
  • IQR: {(len(outliers_ambos)/outliers_iqr_unicos*100 if outliers_iqr_unicos > 0 else 0):.1f}% confirmados por IF
  • IF: {(len(outliers_ambos)/outliers_if_unicos*100 if outliers_if_unicos > 0 else 0):.1f}% confirmados por IQR
"""

ax5.text(0.05, 0.95, comparison_text, transform=ax5.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ============================================================================
# GRÁFICO 6: Scatter 2D - Sepal Length vs Sepal Width
# ============================================================================
ax6 = fig.add_subplot(gs[1, 2])

normal = df[~df['outlier_iqr'] & ~df['outlier_if']]
ax6.scatter(normal[feature_names[0]], normal[feature_names[1]], 
           color='blue', alpha=0.6, s=50, label='Normal')

if len(outliers_both) > 0:
    both = df[df['outlier_iqr'] & df['outlier_if']]
    ax6.scatter(both[feature_names[0]], both[feature_names[1]],
               color='darkred', s=200, marker='*', label='Ambos', edgecolors='black', linewidth=2)

if len(outliers_solo_iqr) > 0:
    solo_iqr = df[df['outlier_iqr'] & ~df['outlier_if']]
    ax6.scatter(solo_iqr[feature_names[0]], solo_iqr[feature_names[1]],
               color='orange', s=150, marker='s', label='Solo IQR', edgecolors='black', linewidth=1.5)

if len(outliers_solo_if) > 0:
    solo_if = df[~df['outlier_iqr'] & df['outlier_if']]
    ax6.scatter(solo_if[feature_names[0]], solo_if[feature_names[1]],
               color='purple', s=150, marker='^', label='Solo IF', edgecolors='black', linewidth=1.5)

ax6.set_xlabel(feature_names[0], fontweight='bold')
ax6.set_ylabel(feature_names[1], fontweight='bold')
ax6.set_title('Plot 2D: Sepal Length vs Width', fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 7: Anomaly Scores Distribution
# ============================================================================
ax7 = fig.add_subplot(gs[2, 0])

normal_scores = df[~df['outlier_if']]['anomaly_score']
outlier_scores = df[df['outlier_if']]['anomaly_score']

ax7.hist(normal_scores, bins=30, color='skyblue', alpha=0.7, label='Normal', edgecolor='black')
ax7.hist(outlier_scores, bins=15, color='red', alpha=0.7, label='Outlier (IF)', edgecolor='darkred')
ax7.axvline(iso_forest.offset_, color='red', linestyle='--', linewidth=2, label='Threshold')
ax7.set_xlabel('Anomaly Score', fontweight='bold')
ax7.set_ylabel('Frecuencia', fontweight='bold')
ax7.set_title('Distribución de Anomaly Scores', fontweight='bold')
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3, axis='y')

# ============================================================================
# GRÁFICO 8: Métrica de Aislamiento
# ============================================================================
ax8 = fig.add_subplot(gs[2, 1])

indices = np.arange(len(df))
colors = ['red' if df.iloc[i]['outlier_if'] else 'blue' for i in range(len(df))]
sizes = [200 if df.iloc[i]['outlier_if'] else 50 for i in range(len(df))]

scatter = ax8.scatter(indices, anomaly_scores, c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=0.5)
ax8.axhline(iso_forest.offset_, color='red', linestyle='--', linewidth=2, label='Threshold')
ax8.set_xlabel('Índice de Muestra', fontweight='bold')
ax8.set_ylabel('Anomaly Score', fontweight='bold')
ax8.set_title('Anomaly Scores por Muestra (Isolation Forest)', fontweight='bold')
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)

# ============================================================================
# GRÁFICO 9: Matriz de Concordancia
# ============================================================================
ax9 = fig.add_subplot(gs[2, 2])

# Crear matriz de confusión
concordancia = np.array([
    [len(df[~df['outlier_iqr'] & ~df['outlier_if']]), len(outliers_solo_iqr)],
    [len(outliers_solo_if), len(outliers_ambos)]
])

im = ax9.imshow(concordancia, cmap='YlOrRd', aspect='auto')
ax9.set_xticks([0, 1])
ax9.set_yticks([0, 1])
ax9.set_xticklabels(['Normal (IF)', 'Outlier (IF)'])
ax9.set_yticklabels(['Normal (IQR)', 'Outlier (IQR)'])

# Anotaciones
for i in range(2):
    for j in range(2):
        text = ax9.text(j, i, concordancia[i, j],
                       ha="center", va="center", color="black", fontsize=14, fontweight='bold')

ax9.set_title('Matriz de Concordancia\n(IQR vs IF)', fontweight='bold')

# ============================================================================
# GUARDAR GRÁFICO PRINCIPAL
# ============================================================================

plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/comparativa_outliers.png',
            dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: comparativa_outliers.png")
plt.close()

# ============================================================================
# 7. GRÁFICO ADICIONAL: Heatmap de Características por Outlier
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Normalizar datos para heatmap
X_normalized = StandardScaler().fit_transform(X)

# Outliers IQR
if len(outliers_solo_iqr) > 0:
    outliers_data_iqr = X_normalized[list(outliers_solo_iqr)]
    sns.heatmap(outliers_data_iqr, annot=True, fmt='.2f', cmap='RdYlBu_r',
               xticklabels=feature_names, yticklabels=[f'Idx {i}' for i in sorted(outliers_solo_iqr)],
               ax=axes[0], cbar_kws={'label': 'Valor Normalizado'})
    axes[0].set_title(f'Características: Outliers Solo IQR ({len(outliers_solo_iqr)})', fontweight='bold')
else:
    axes[0].text(0.5, 0.5, 'No hay outliers\nsolamente en IQR', 
                ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
    axes[0].set_title('Características: Outliers Solo IQR (0)', fontweight='bold')

# Outliers IF
if len(outliers_solo_if) > 0:
    outliers_data_if = X_normalized[list(outliers_solo_if)]
    sns.heatmap(outliers_data_if, annot=True, fmt='.2f', cmap='RdYlGn_r',
               xticklabels=feature_names, yticklabels=[f'Idx {i}' for i in sorted(outliers_solo_if)],
               ax=axes[1], cbar_kws={'label': 'Valor Normalizado'})
    axes[1].set_title(f'Características: Outliers Solo IF ({len(outliers_solo_if)})', fontweight='bold')
else:
    axes[1].text(0.5, 0.5, 'No hay outliers\nsolamente en IF', 
                ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    axes[1].set_title('Características: Outliers Solo IF (0)', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/alepulsito/projects/hybridge/analisis-y-procesamiento-de-datos/hito-2/heatmap_outliers.png',
            dpi=300, bbox_inches='tight')
print(f"✅ Gráfico guardado: heatmap_outliers.png")
plt.close()

# ============================================================================
# 8. TABLA DE RESULTADOS
# ============================================================================

print(f"\n" + "=" * 80)
print("TABLA DE RESULTADOS FINAL")
print("=" * 80)

results_df = pd.DataFrame({
    'Método': ['IQR (Univariado)', 'Isolation Forest (Multivariado)', 'Solapamiento'],
    'Outliers Detectados': [outliers_iqr_unicos, outliers_if_unicos, len(outliers_ambos)],
    'Únicos': [len(outliers_solo_iqr), len(outliers_solo_if), '-']
})

print(f"\n{results_df.to_string(index=False)}")

# ============================================================================
# 9. CONCLUSIONES
# ============================================================================

print(f"\n" + "=" * 80)
print("CONCLUSIONES Y OBSERVACIONES")
print("=" * 80)

print(f"""
🔍 DIFERENCIAS CLAVE:

IQR (Univariado):
  ✓ Detecta valores extremos en CADA variable individual
  ✓ Rápido y fácil de interpretar
  ✗ No considera RELACIONES entre variables
  ✗ Puede pasar por alto anomalías multivariadas

Isolation Forest (Multivariado):
  ✓ Considera TODAS las variables simultáneamente
  ✓ Detecta anomalías en el espacio multidimensional
  ✓ Identifica patrones anómalos complejos
  ✗ Más computacionalmente intensivo
  ✗ Requiere ajuste de parámetros (contamination)

📊 RESULTADO EN ESTE DATASET:
  • Outliers solo detectedos por IQR: {len(outliers_solo_iqr)}
  • Outliers solo detectados por IF: {len(outliers_solo_if)}
  • Concordancia: {(len(outliers_ambos)/(len(outliers_iqr_indices) if outliers_iqr_indices else 1)*100):.1f}%

💡 RECOMENDACIÓN:
  Usar AMBOS métodos para una detección más robusta:
  - IQR para anomalías univariadas claras
  - Isolation Forest para anomalías complejas multivariadas
  - Triangular con un tercer método si es crítico
""")

print(f"\n" + "=" * 80)
