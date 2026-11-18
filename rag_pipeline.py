"""
rag_pipeline.py - Motor de Recuperación y Generación Aumentada (RAG) con LCEL y Ollama.

Arquitectura:
    - Embeddings: `OllamaEmbeddings(model="bge-m3")` de `langchain_ollama`
    - Vector Store: `Chroma` de `langchain_chroma` persistido en `./chroma_db`
    - LLM Generador: `ChatOllama(model="granite4:latest", temperature=0.1)` de `langchain_ollama`
    - Prompting: `ChatPromptTemplate` de `langchain_core.prompts` con directiva estricta
      de abstención literal ante consultas sin sustento en el corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ingesta import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

# Configuración del modelo generativo y de recuperación
LLM_MODEL = "granite4:latest"
LLM_TEMPERATURE = 0.1
TOP_K = 4

# Frase canónica de abstención exigida por el diseño del sistema
MENSAJE_ABSTENCION = (
    "No hay evidencia suficiente en el recetario para elaborar una receta con esos ingredientes."
)

# Plantilla de prompt con rol de sistema y mensaje humano (LangChain Core)
SYSTEM_PROMPT = """Sos un asistente culinario técnico y riguroso basado exclusivamente en el recetario verificado de Paulina Cocina disponible en el CONTEXTO RECUPERADO.

REGLAS ESTRICTAS DE COMPORTAMIENTO (OBLIGATORIAS):
1. Respondé ÚNICAMENTE utilizando la información presente de forma explícita en el CONTEXTO RECUPERADO que se te provee abajo.
2. NO uses tu conocimiento general preentrenado para inventar recetas, ingredientes, proporciones, reemplazos ni pasos de cocción que no figuren en el contexto.
3. Si el usuario consulta por un plato o por ingredientes principales que NO corresponden a las recetas del contexto (el corpus solo contiene: Guiso de lentejas, Tortilla de papas española y Empanadas de carne), debés abstenerte de responder y devolver EXACTAMENTE esta frase literal sin agregar recetas inventadas:
"No hay evidencia suficiente en el recetario para elaborar una receta con esos ingredientes."
4. Cuando la consulta SÍ pueda responderse con el contexto:
   - Indicá claramente el nombre de la receta correspondiente.
   - Detallá los ingredientes o las instrucciones paso a paso tal como figuran en la evidencia.
   - Citá siempre entre corchetes el archivo fuente utilizado (por ejemplo: `[Fuente: receta_01_guiso_lentejas.txt]`)."""

HUMAN_PROMPT = """=== CONTEXTO RECUPERADO DEL RECETARIO ===
{contexto}

=== CONSULTA O INGREDIENTES DEL USUARIO ===
{consulta}

Recordá: si los ingredientes o el plato consultado no pertenecen a las recetas del contexto recuperado, respondé exactamente:
"No hay evidencia suficiente en el recetario para elaborar una receta con esos ingredientes."
Respuesta:"""


def obtener_vectorstore(directorio_chroma: Path = CHROMA_DIR) -> Chroma:
    """
    Conecta con la base vectorial ChromaDB persistida en `./chroma_db`
    usando `OllamaEmbeddings(model="bge-m3")`.
    """
    if not directorio_chroma.exists() or not any(directorio_chroma.iterdir()):
        raise FileNotFoundError(
            f"La base vectorial en '{directorio_chroma}' no existe o está vacía. "
            "Ejecutá primero `python ingesta.py` para construir el índice."
        )

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_gpu=0)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(directorio_chroma),
    )


def obtener_retriever(k: int = TOP_K):
    """Devuelve el objeto Retriever estándar de LangChain Chroma configurado con `k`."""
    vectorstore = obtener_vectorstore()
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


def recuperar_fragmentos(consulta: str, k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Recupera los `k` fragmentos más relevantes desde ChromaDB junto con su
    distancia vectorial (L2/coseno según configuración de Chroma) y metadatos completos.
    """
    vectorstore = obtener_vectorstore()
    resultados_con_score = vectorstore.similarity_search_with_score(consulta, k=k)

    fragmentos_formateados: list[dict[str, Any]] = []
    for idx, (doc, score) in enumerate(resultados_con_score, start=1):
        archivo = doc.metadata.get(
            "archivo", Path(doc.metadata.get("source", "desconocido.txt")).name
        )
        fragmentos_formateados.append(
            {
                "ranking": idx,
                "texto": doc.page_content,
                "archivo": archivo,
                "receta": doc.metadata.get("receta", "Sin título"),
                "url_origen": doc.metadata.get("url_origen", "https://www.paulinacocina.net"),
                "autor": doc.metadata.get("autor", "Paulina Cocina"),
                "fecha_captura": doc.metadata.get("fecha_captura", "N/D"),
                "chunk_id": doc.metadata.get("chunk_id", idx),
                "score": round(float(score), 4),
                "metadatos": doc.metadata,
            }
        )

    return fragmentos_formateados


def formatear_contexto_para_prompt(fragmentos: list[dict[str, Any]]) -> str:
    """Convierte la lista de fragmentos recuperados en un bloque textual trazable para el prompt."""
    if not fragmentos:
        return "SIN FRAGMENTOS RECUPERADOS."

    bloques: list[str] = []
    for frag in fragmentos:
        encabezado = (
            f"--- FRAGMENTO #{frag['ranking']} | "
            f"Archivo: {frag['archivo']} (Chunk #{frag['chunk_id']}) | "
            f"Receta: {frag['receta']} | "
            f"Distancia vectorial: {frag['score']:.4f} ---"
        )
        bloques.append(f"{encabezado}\n{frag['texto']}")

    return "\n\n".join(bloques)


def construir_cadena_generacion():
    """
    Construye la cadena LCEL (`ChatPromptTemplate | ChatOllama | StrOutputParser`)
    sin utilizar clases deprecadas como `RetrievalQA`.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        num_gpu=0,
    )
    return prompt | llm | StrOutputParser()


def responder(consulta: str, k: int = TOP_K) -> tuple[str, list[dict[str, Any]]]:
    """
    Ejecuta el flujo completo RAG para una consulta de usuario:
    1. Valida que la consulta no esté vacía.
    2. Recupera los `k` fragmentos más cercanos de `./chroma_db` con `bge-m3`.
    3. Invoca el generador local `ChatOllama(model="granite4:latest", temperature=0.1)`.
    4. Devuelve la tupla `(respuesta_texto, lista_de_fragmentos_recuperados)`.
    """
    consulta_limpia = (consulta or "").strip()
    if not consulta_limpia:
        return (
            "Por favor, ingresá al menos un ingrediente o una consulta culinaria para buscar en el recetario.",
            [],
        )

    # 1. Recuperación semántica en ChromaDB
    fragmentos = recuperar_fragmentos(consulta_limpia, k=k)
    if not fragmentos:
        return MENSAJE_ABSTENCION, []

    # 2. Formateo del contexto con trazabilidad documental
    contexto_str = formatear_contexto_para_prompt(fragmentos)

    # 3. Generación controlada mediante LCEL y ChatOllama
    cadena = construir_cadena_generacion()
    respuesta_texto = cadena.invoke(
        {
            "contexto": contexto_str,
            "consulta": consulta_limpia,
        }
    ).strip()

    return respuesta_texto, fragmentos


if __name__ == "__main__":
    # Prueba rápida por consola del pipeline RAG
    pregunta_demo = "Tengo lentejas, chorizo colorado, panceta, papa y puré de tomate. ¿Qué puedo cocinar?"
    print(f"Consulta de prueba: {pregunta_demo}\n")
    texto_resp, frags = responder(pregunta_demo)
    print("=== RESPUESTA GENERADA ===")
    print(texto_resp)
    print("\n=== FRAGMENTOS RECUPERADOS ===")
    for f in frags:
        print(
            f"- [{f['archivo']} | Chunk #{f['chunk_id']} | dist={f['score']}] "
            f"{f['texto'][:100]}..."
        )
