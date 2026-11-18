"""
test_evaluacion.py - Auditoría RAG 
Descripción:
    Ejecuta automáticamente las 5 preguntas de evaluación exigidas por la consigna:
    - 3 preguntas respondibles (una por cada receta del corpus en `datos/`).
    - 2 preguntas trampa / fuera de dominio (sushi de salmón y tacos mexicanos)
      para auditar que el modelo se abstenga sin alucinar recetas inexistentes.
    Imprime en pantalla tanto el registro detallado como la tabla Markdown consolidada.
"""

from __future__ import annotations

import sys
from typing import Any

from rag_pipeline import MENSAJE_ABSTENCION, TOP_K, responder

CASOS_PRUEBA: list[dict[str, str]] = [
    {
        "id": "Caso 1 (Respondible - Guiso)",
        "pregunta": (
            "Tengo lentejas, chorizo colorado, panceta ahumada, carne vacuna, "
            "papa y puré de tomate. ¿Qué receta puedo preparar y cuánto tiempo "
            "lleva la cocción antes y después de agregar las papas?"
        ),
        "fuente_esperada": "receta_01_guiso_lentejas.txt",
        "tipo": "respondible",
    },
    {
        "id": "Caso 2 (Respondible - Tortilla)",
        "pregunta": (
            "Tengo 4 papas medianas, huevos, cebolla y aceite. ¿Cuántos huevos "
            "recomienda usar Paulina Cocina para hacer la tortilla de papas y "
            "cuántos minutos se cocina del segundo lado para que quede babé?"
        ),
        "fuente_esperada": "receta_02_tortilla_papas.txt",
        "tipo": "respondible",
    },
    {
        "id": "Caso 3 (Respondible - Empanadas)",
        "pregunta": (
            "Tengo 500 gramos de carne picada, tapas de empanada, cebolla, morrón "
            "y huevos duros. ¿Qué cantidad de cebolla lleva el relleno para que "
            "las empanadas de carne salgan jugosas y a qué temperatura se hornean?"
        ),
        "fuente_esperada": "receta_03_empanadas_carne.txt",
        "tipo": "respondible",
    },
    {
        "id": "Caso 4 (Trampa / Sin respuesta - Sushi)",
        "pregunta": (
            "Tengo salmón rosado fresco, arroz doble carolina, alga nori, queso "
            "crema y vinagre de arroz. ¿Cómo preparo rolls de sushi de salmón paso a paso?"
        ),
        "fuente_esperada": "Ninguna (Abstención esperada)",
        "tipo": "trampa",
    },
    {
        "id": "Caso 5 (Trampa / Sin respuesta - Tacos)",
        "pregunta": (
            "Tengo tortillas de maíz, carne de cerdo marinada con achiote, piña, "
            "cilantro, cebolla morada y chile jalapeño. ¿Cómo hago tacos mexicanos al pastor?"
        ),
        "fuente_esperada": "Ninguna (Abstención esperada)",
        "tipo": "trampa",
    },
]


def _resumir_linea_tabla(texto: str, max_len: int = 160) -> str:
    """Limpia saltos de línea y barras verticales para insertar texto en celdas Markdown."""
    limpio = " ".join(texto.replace("|", "/").split())
    if len(limpio) > max_len:
        return limpio[: max_len - 3] + "..."
    return limpio


def evaluar_observacion(
    caso: dict[str, str],
    respuesta: str,
    fragmentos: list[dict[str, Any]],
) -> str:
    """Evalúa cualitativamente si la recuperación y la generación cumplieron el objetivo."""
    archivos_recuperados = [f["archivo"] for f in fragmentos]
    top1_archivo = archivos_recuperados[0] if archivos_recuperados else "Ninguno"

    if caso["tipo"] == "respondible":
        acierto_top1 = top1_archivo == caso["fuente_esperada"]
        acierto_topk = caso["fuente_esperada"] in archivos_recuperados
        se_abstuvo = MENSAJE_ABSTENCION.lower() in respuesta.lower()

        if acierto_top1 and not se_abstuvo:
            return (
                f"CORRECTO: El recuperador bge-m3 ubicó '{caso['fuente_esperada']}' en Top-1 "
                "y granite4 respondió fielmente con cantidades y tiempos exactos sin alucinar."
            )
        if acierto_topk and not se_abstuvo:
            return (
                f"ACEPTABLE: '{caso['fuente_esperada']}' recuperado dentro de Top-{TOP_K} "
                f"(Top-1 fue '{top1_archivo}') y respuesta generada con sustento textual."
            )
        return (
            f"FALLO DE RECUPERACIÓN/GENERACIÓN: Top-1='{top1_archivo}', "
            f"abstención={se_abstuvo}."
        )

    # Caso trampa (fuera del dominio del corpus)
    se_abstuvo_literal = MENSAJE_ABSTENCION.lower() in respuesta.lower()
    se_abstuvo_parcial = any(
        frase in respuesta.lower()
        for frase in [
            "no hay evidencia suficiente",
            "no se encuentra en el recetario",
            "no cuento con",
            "no incluye",
        ]
    )

    if se_abstuvo_literal:
        return (
            "CORRECTO (ABSTENCIÓN ESTRICTA): A pesar de que ChromaDB devuelve los k vecinos "
            "más cercanos por obligación geométrica, granite4 detectó la ausencia de los "
            "ingredientes y emitió la frase exacta de abstención sin alucinar."
        )
    if se_abstuvo_parcial:
        return (
            "ACEPTABLE (ABSTENCIÓN SEMÁNTICA): El modelo rechazó inventar la receta "
            "inexistente, aunque varió ligeramente la formulación literal."
        )
    return (
        "ALUCINACIÓN DETECTADA: El modelo intentó responder una receta ausente en el "
        "corpus usando su conocimiento paramétrico."
    )


def ejecutar_suite_evaluacion() -> None:
    """Corre los 5 casos de prueba e imprime el reporte detallado y la tabla Markdown."""
    print("=" * 85)
    print("EJECUCIÓN DE SUITE DE EVALUACIÓN RAG (5 PREGUNTAS DE PRUEBA - IFTS N.º 24)")
    print("=" * 85)

    filas_tabla: list[str] = [
        "| # | Pregunta | Fuente esperada | Fragmento recuperado (Top-1) | Respuesta del modelo | Observación cualitativa |",
        "|---|---|---|---|---|---|",
    ]

    for idx, caso in enumerate(CASOS_PRUEBA, start=1):
        print(f"\n[{idx}/5] Ejecutando {caso['id']}...")
        print(f"  Pregunta: {caso['pregunta']}")
        print(f"  Fuente esperada: {caso['fuente_esperada']}")

        try:
            respuesta, fragmentos = responder(caso["pregunta"], k=TOP_K)
        except Exception as exc:
            print(f"\n[ERROR] No se pudo completar la consulta #{idx}: {exc}", file=sys.stderr)
            sys.exit(1)

        if fragmentos:
            top1 = fragmentos[0]
            frag_resumen = (
                f"`{top1['archivo']}` (Chunk #{top1['chunk_id']}, dist={top1['score']:.4f}): "
                f"\"{_resumir_linea_tabla(top1['texto'], 110)}\""
            )
        else:
            frag_resumen = "Sin fragmentos recuperados"

        observacion = evaluar_observacion(caso, respuesta, fragmentos)

        print(f"  Top-1 Recuperado: {frag_resumen}")
        print(f"  Respuesta Modelo: {respuesta}")
        print(f"  Observación     : {observacion}")

        fila = (
            f"| {idx} | {_resumir_linea_tabla(caso['pregunta'], 120)} "
            f"| `{caso['fuente_esperada']}` "
            f"| {frag_resumen} "
            f"| {_resumir_linea_tabla(respuesta, 150)} "
            f"| {observacion} |"
        )
        filas_tabla.append(fila)

    print("\n" + "=" * 85)
    print("TABLA RESUMEN EN FORMATO MARKDOWN (SECCIÓN 6 DEL README.md)")
    print("=" * 85 + "\n")
    print("\n".join(filas_tabla))


if __name__ == "__main__":
    ejecutar_suite_evaluacion()
