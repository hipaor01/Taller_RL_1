# Talleres de aprendizaje por refuerzo: trading simulado

Este repositorio reúne dos talleres de aprendizaje por refuerzo (RL) aplicados a un entorno sencillo de trading. El objetivo es estudiar cómo aprende un agente a tomar decisiones y comparar sus políticas en distintos mercados **sintéticos**. Los notebooks contienen el enunciado, las implementaciones, las búsquedas de hiperparámetros y el análisis de los experimentos.

## El problema

En cada episodio, el agente observa una serie de precios de un solo activo y decide **vender, mantener o comprar**. El estado original resume la tendencia reciente del precio y si el agente tiene o no el activo. Parte de un capital de 1.000 unidades y paga una comisión del 0,1 % por operación efectiva. La recompensa incorpora el cambio relativo del precio mientras mantiene el activo y el coste de las operaciones; el retorno final se calcula a partir del balance de la cartera.

Se entrenan y evalúan agentes en cuatro generadores de precios:

- **Tendencia y ciclo:** combina una tendencia suave con una componente cíclica.
- **Movimiento browniano:** genera una evolución aleatoria de precios.
- **Reversión a la media:** el precio tiende a volver hacia un nivel de referencia.
- **Cambios de régimen:** alterna periodos alcistas y bajistas.

## Algoritmos y notebooks

| Taller | Trabajo | Notebook |
| --- | --- | --- |
| 1 | Q-learning tabular, selección de hiperparámetros y evaluación | [Q-learning — Higinio](taller_trader_higinio.ipynb) |
| 1 | SARSA tabular, selección de hiperparámetros y evaluación | [SARSA — Javi](taller_trader_javi.ipynb) |
| 1 | Comparación de Q-learning y SARSA con test común y estado enriquecido | [Experimento de representación — Hugo](taller_trader_hugo.ipynb) |
| 2 | Deep Q-Network (DQN) | [DQN — Javi](taller_trader2_javi.ipynb) |
| 2 | Asynchronous Advantage Actor-Critic (A3C) | [A3C — Higinio](taller_trader2_higinio.ipynb) |
| 2 | Proximal Policy Optimization (PPO) y estado enriquecido | [PPO — Hugo](taller_trader2_hugo.ipynb) |

**Q-learning** aprende una tabla de valores para parejas de estado y acción. Su actualización usa el mayor valor estimado del siguiente estado, aunque durante el entrenamiento se explore con otra acción. Es un método *off-policy*.

**SARSA** también aprende una tabla, pero actualiza el valor usando la siguiente acción elegida por la propia política de entrenamiento. Es *on-policy*: su actualización refleja también la exploración. Ambos métodos tabulares necesitan discretizar el estado.

**DQN** sustituye la tabla por una red neuronal que aproxima los valores de las acciones. Utiliza experiencias guardadas y una red objetivo para entrenar el estimador; así se puede trabajar con observaciones sin construir una tabla para cada estado.

**A3C** combina un *actor*, que aprende una política de acciones, y un *crítico*, que estima el valor de los estados. Varios trabajadores generan experiencias y actualizan un modelo compartido de forma asíncrona.

**PPO** también utiliza actor y crítico, pero limita los cambios de política mediante un objetivo recortado (*clipping*). El notebook estudia tanto el estado original como una variante enriquecida.

## Instalación y primeros pasos

Se necesita Python 3.10 o posterior. Una forma de preparar el entorno es con [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra notebook --locked
uv run jupyter lab
```

Abre cualquiera de los notebooks desde JupyterLab. Sus salidas guardadas permiten consultar los experimentos sin volver a entrenar. Algunos notebooks de Hugo también usan `pandas` y `scipy`, que no están declarados en `pyproject.toml`; para abrir Jupyter con esas dependencias disponibles en la sesión:

```bash
uv run --extra notebook --with pandas --with scipy jupyter lab
```

Estas dos dependencias adicionales no están fijadas en `uv.lock`; registra sus versiones si necesitas repetir esos experimentos. Antes de ejecutar todas las celdas, revisa las secciones de búsqueda: entrenan muchas configuraciones y semillas y pueden tardar considerablemente. Los tiempos dependen del equipo. La comparación tabular de Hugo indica aproximadamente 25 minutos en un portátil; las búsquedas completas del segundo taller tienen presupuestos mayores.

## Evaluación y reproducibilidad

En los experimentos se generan series diferentes para entrenar y se reserva validación para elegir configuraciones. Una vez fijada la elección, el test permite evaluar en series no usadas para seleccionarla. Las políticas se comparan con referencias como permanecer en efectivo, actuar aleatoriamente y comprar y mantener (*buy & hold*); las métricas y referencias concretas se detallan en cada notebook.

Para el **segundo taller**, el protocolo común fija las semillas de entrenamiento `101, 202, 303, 404, 505`, 50 series de validación y 200 de test por generador, con 500 precios por serie. Los bancos de evaluación tienen semillas independientes y se comprueban mediante huellas. A3C puede variar entre ejecuciones aun con las mismas semillas, porque el orden de actualización de los trabajadores asíncronos depende de su planificación.

El primer taller incluye tanto evaluaciones propias de cada notebook como un experimento posterior con test común. Consulta la configuración y el protocolo concreto antes de comparar cifras: no todos los experimentos usan las mismas semillas, presupuestos de entrenamiento o búsquedas. Las evaluaciones de varias semillas sobre las mismas series comparten mercados y no representan series independientes adicionales.

Las conclusiones describen este entorno simulado. En particular, el generador de tendencia y ciclo produce retornos extremos, y un criterio heurístico de estabilidad no demuestra convergencia ni garantiza una política óptima. Los resultados no constituyen evidencia de rendimiento en mercados reales.

## Scripts y artefactos

- [run_a3c_search.py](run_a3c_search.py) ejecuta o reanuda la búsqueda A3C definida en el notebook de Higinio. [run_a3c_training_length.py](run_a3c_training_length.py) y [run_a3c_final_test.py](run_a3c_final_test.py) ejecutan los análisis de duración y test final, respectivamente. Dependen de celdas y resultados definidos en ese notebook.
- [ppo_core.py](ppo_core.py) contiene el entorno, protocolo y componentes PPO reutilizados por [run_ppo_search.py](run_ppo_search.py). Por ejemplo, `uv run python run_ppo_search.py reversion_media original` inicia la búsqueda y evaluación para ese mercado y estado.
- Los scripts generan resultados en `artifacts/`, directorio ignorado por Git. Los notebooks pueden mostrar salidas ya guardadas aunque los modelos o archivos intermedios locales no estén incluidos en el repositorio. Revisa los requisitos y el coste de cada experimento antes de ejecutarlo.
