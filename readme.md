# Detección de Riesgos de Salud Mental en Redes Sociales

##  Definición del problema

Este proyecto se enfoca en un problema delicado pero de gran relevancia: **detectar señales tempranas de riesgo de salud mental en redes sociales**. La idea es entrenar un modelo que pueda leer el contenido de una publicación (específicamente, un tuit) y decir si muestra signos de angustia emocional (**distress**) o si es una publicación neutral o positiva (**no distress**).

Aunque existen ya iniciativas de intervención psicológica en línea, **automatizar la detección inicial** puede hacer una gran diferencia, especialmente en entornos donde hay millones de publicaciones diarias y los equipos humanos no dan abasto. Separar lo preocupante de lo que no lo es permite priorizar mejor, ofrecer recursos de ayuda más rápido y sentar una base sólida para modelos más avanzados, como detección de urgencia o personalización de intervenciones.

Este proyecto es una forma práctica de trabajar con **texto real, emocional y contextual**, aplicando técnicas modernas de procesamiento de lenguaje natural (NLP) sobre datos ya etiquetados.

---

## Experiencia

Para este proyecto se usará el [**SMILE Twitter Emotion Dataset**](https://figshare.com/articles/dataset/smile_annotations_final_csv/3187909), desarrollado como parte del proyecto de investigación [CultureSmile](http://www.culturesmile.org). Este dataset fue publicado en conjunto con el paper académico:

> Wang et al., *SMILE: Twitter Emotion Classification using Domain Adaptation*, 2016.  
> https://www.zubiaga.org/publications/files/wang-2016-saaip.pdf

El dataset contiene **3,085 tuits recolectados entre 2013 y 2015** en interacciones con cuentas de museos británicos. Cada tuit ha sido anotado manualmente con una de cinco emociones:

- `anger`
- `disgust`
- `sadness`
- `happiness`
- `surprise`

### Etiquetado binario para este proyecto:

Se agrupan las emociones en dos clases para simplificar el problema a una clasificación binaria:

| Emociones originales | Clase final |
|----------------------|-------------|
| anger, disgust, sadness | **Distress** (1) |
| happiness, surprise     | **No distress** (0) |

Esto permite construir un modelo que aprenda a detectar publicaciones con carga emocional negativa, potencialmente indicadoras de angustia psicológica.

**Ventajas del dataset:**

-  Ya viene anotado y balanceado en varios tipos de emoción.
-  Anotaciones de alta calidad con base en un protocolo académico.
-  Datos públicos y accesibles sin necesidad de scraping.
-  Lista para ser procesada en formato CSV y compatible con pipelines modernos.

El plan es aplicar técnicas como limpieza de texto, tokenización, y representaciones vectoriales (TF-IDF, embeddings BERT) para convertir el contenido en entradas útiles para distintos modelos.

Luego se probarán varios clasificadores como:

- **Naive Bayes** (eficiente y simple)
- **Regresión logística** (interpretable)
- **Random Forest o XGBoost** (robustos con features tabulares)
- **Transformers como BERT** (modelo contextual de última generación)

Esto permite comenzar con modelos simples e ir aumentando la complejidad de manera progresiva.

---

##  Evaluación del rendimiento

Como se trata de un problema de clasificación binaria (*distress vs no distress*), las métricas elegidas serán:

- **Recall (sensibilidad)**: para medir qué tan bien el modelo detecta los mensajes con riesgo.
- **Precisión**: para evitar emitir demasiadas alertas falsas.
- **F1-score**: que equilibra ambos valores cuando las clases están desbalanceadas.
- **Matriz de confusión**: para identificar cuántos tuits se clasifican erróneamente, y de qué forma (por ejemplo, si un tuit de angustia se clasifica como no-risk).
- **Área bajo la curva Precision-Recall (AU PR)**: especialmente útil cuando la clase de interés es menos frecuente.

El foco estará en construir un modelo que **minimice los falsos negativos**, es decir, que no deje pasar tuits de distress sin detectar, pero que al mismo tiempo **no genere demasiadas falsas alarmas** que puedan saturar los canales de ayuda.

Con este enfoque se busca **no solo resolver el problema de detección de angustia en redes sociales**, sino también sentar las bases para futuras tareas como:

- Clasificación por nivel de urgencia  
- Priorización de casos en centros de ayuda  
- Análisis de evolución emocional a lo largo del tiempo  
- Recomendación personalizada de recursos psicológicos

---

## Dataset utilizado

- **Nombre**: SMILE Twitter Emotion Dataset  
- **Autores**: Wang et al. (2016)  
- **Licencia**: Anotaciones bajo CC-BY; contenido de Twitter sujeto a sus Términos de Servicio  
- **Descarga**: https://figshare.com/articles/dataset/smile_annotations_final_csv/3187909
