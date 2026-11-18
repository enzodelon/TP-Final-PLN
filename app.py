"""
app.py - Interfaz gráfica interactiva local en Gradio para el sistema RAG culinario.

Ejecución:
    python app.py
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from rag_pipeline import EMBEDDING_MODEL, LLM_MODEL, TOP_K, responder

FORMULA_RECTORA = (
    "Este sistema permite que una persona que cocina en su casa ingrese los "
    "ingredientes que tiene disponibles para consultar recetas viables en un "
    "corpus de recetas reales extraídas de Paulina Cocina (www.paulinacocina.net) "
    "y obtener instrucciones paso a paso con evidencia textual comprobable y "
    "trazable a la fuente original."
)

EJEMPLOS_CONSULTAS: list[list[str]] = [
    [
        "Tengo lentejas, chorizo colorado, panceta ahumada, carne, papa y puré de tomate. ¿Qué receta puedo preparar y cuáles son los pasos?"
    ],
    [
        "Tengo 4 papas, huevos, cebolla y aceite. ¿Cuál es la proporción de huevos que recomienda Paulina Cocina para la tortilla de papas y cómo logro que quede babé?"
    ],
    [
        "Quiero hacer empanadas de carne bien jugosas. ¿Qué cantidad de cebolla lleva en relación a los 500 g de carne y a qué temperatura se hornean?"
    ],
    [
        "Tengo salmón rosado fresco, arroz doble carolina, alga nori y vinagre de arroz. ¿Cómo preparo sushi de salmón paso a paso?"
    ],
    [
        "Tengo tortillas de maíz, palta, cilantro, lima y chile jalapeño. ¿Me das la receta para hacer tacos mexicanos al pastor?"
    ],
]


def formatear_panel_evidencia(fragmentos: list[dict[str, Any]]) -> str:
    """
    Construye el panel Markdown de evidencia documental mostrando el texto exacto
    de cada fragmento recuperado por `bge-m3`, su archivo fuente, URL y distancia vectorial.
    """
    if not fragmentos:
        return (
            "⚠️ **No se recuperaron fragmentos del índice vectorial.**\n\n"
            "Verificá haber ingresado una consulta válida y haber ejecutado `python ingesta.py`."
        )

    bloques_md: list[str] = [
        f"### 📚 Fragmentos recuperados desde ChromaDB (`{EMBEDDING_MODEL}`, `k={len(fragmentos)}`)\n"
    ]

    for frag in fragmentos:
        bloque = (
            f"#### 📄 Fragmento #{frag['ranking']} — `{frag['archivo']}` "
            f"(Chunk #{frag['chunk_id']} | Distancia: `{frag['score']:.4f}`)\n"
            f"- **Receta:** {frag['receta']}\n"
            f"- **Autor:** {frag['autor']} | **Captura:** {frag['fecha_captura']}\n"
            f"- **URL de origen:** {frag['url_origen']}\n\n"
            f"```text\n{frag['texto']}\n```\n"
            "---"
        )
        bloques_md.append(bloque)

    return "\n\n".join(bloques_md)


def procesar_consulta_ui(consulta: str) -> tuple[str, str]:
    """
    Callback de Gradio que conecta la consulta del usuario con `rag_pipeline.responder`
    y devuelve las dos salidas separadas: (respuesta_generada_md, evidencia_recuperada_md).
    """
    try:
        respuesta_texto, fragmentos = responder(consulta, k=TOP_K)
        evidencia_md = formatear_panel_evidencia(fragmentos)
        return respuesta_texto, evidencia_md
    except Exception as exc:
        mensaje_error = (
            f"❌ **Error al ejecutar el pipeline RAG:** `{exc}`\n\n"
            "**Pasos de verificación:**\n"
            "1. Asegurate de haber ejecutado `python ingesta.py` para crear `./chroma_db`.\n"
            "2. Verificá que el servicio Ollama esté activo (`ollama serve`).\n"
            f"3. Confirmá tener descargados los modelos: `ollama pull {EMBEDDING_MODEL}` y `ollama pull {LLM_MODEL}`."
        )
        return mensaje_error, "⚠️ Sin evidencia disponible debido a un error de conexión o indexación."


def construir_interfaz() -> gr.Blocks:
    """Construye y retorna la aplicación `gr.Blocks` con diseño estructurado."""
    with gr.Blocks(
        title="RAG Culinario - Paulina Cocina (IFTS N.º 24)"
    ) as demo:
        gr.Markdown(
            f"""
            # 🍲 Asistente RAG Culinario Reproducible — Recetario de Paulina Cocina
            **Tecnicatura Superior en Ciencia de Datos e Inteligencia Artificial (IFTS N.º 24)**
            *Trabajo Integrador 2 — Procesamiento del Lenguaje Natural (Cátedra: Prof. Matías Barreto)*
            *Alumno: Enzo Matías Delon Perez*

            > **Fórmula rectora:** *{FORMULA_RECTORA}*

            **Arquitectura activa:** Embeddings `OllamaEmbeddings(model="{EMBEDDING_MODEL}")` · Base vectorial local `ChromaDB (./chroma_db)` · Generador `ChatOllama(model="{LLM_MODEL}", temperature=0.1)` · Recuperación `Top-K = {TOP_K}`.
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                caja_consulta = gr.Textbox(
                    label="🥕 Ingredientes disponibles o consulta culinaria",
                    placeholder=(
                        "Ej.: Tengo lentejas, chorizo colorado, panceta, cebolla y papas. "
                        "¿Qué receta puedo cocinar paso a paso?"
                    ),
                    lines=3,
                )
            with gr.Column(scale=1):
                boton_consultar = gr.Button(
                    "🔍 Consultar Recetario",
                    variant="primary",
                )
                boton_limpiar = gr.Button("🧹 Limpiar")

        gr.Examples(
            examples=EJEMPLOS_CONSULTAS,
            inputs=caja_consulta,
            label="💡 Consultas de prueba (3 respondibles del corpus + 2 preguntas trampa para auditar abstención)",
        )

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"## 🤖 1. Respuesta Generada (`{LLM_MODEL}`)")
                salida_respuesta = gr.Markdown(
                    value="*Ingresá tus ingredientes y presioná **Consultar Recetario** para generar la respuesta basada en evidencia.*"
                )

            with gr.Column(scale=1):
                gr.Markdown(
                    f"## 🔎 2. Panel de Evidencia / Fuentes Recuperadas (`{EMBEDDING_MODEL}`)"
                )
                salida_evidencia = gr.Markdown(
                    value="*Aquí se mostrarán los fragmentos exactos recuperados desde `./chroma_db` con su archivo fuente y puntaje de distancia.*"
                )

        # Eventos interactivos
        boton_consultar.click(
            fn=procesar_consulta_ui,
            inputs=caja_consulta,
            outputs=[salida_respuesta, salida_evidencia],
        )
        caja_consulta.submit(
            fn=procesar_consulta_ui,
            inputs=caja_consulta,
            outputs=[salida_respuesta, salida_evidencia],
        )
        boton_limpiar.click(
            fn=lambda: (
                "",
                "*Ingresá tus ingredientes y presioná **Consultar Recetario** para generar la respuesta basada en evidencia.*",
                "*Aquí se mostrarán los fragmentos exactos recuperados desde `./chroma_db` con su archivo fuente y puntaje de distancia.*",
            ),
            inputs=None,
            outputs=[caja_consulta, salida_respuesta, salida_evidencia],
        )

    return demo


if __name__ == "__main__":
    interfaz = construir_interfaz()
    interfaz.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
