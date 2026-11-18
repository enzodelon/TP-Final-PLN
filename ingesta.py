"""
ingesta.py - Pipeline de ingesta, enriquecimiento de metadatos, chunking e indexación vectorial.


Justificación analítica de hiperparámetros de segmentación (Chunking):
    - chunk_size = 400 caracteres:
      En textos culinarios estructurados (recetas de Paulina Cocina), cada bloque
      lógico —como la lista compacta de ingredientes o un paso individual de cocción
      ("Paso 1", "Paso 2", etc.)— tiene una extensión promedio de entre 280 y 380
      caracteres. Un tamaño de 400 caracteres permite capturar un paso de preparación
      completo o el núcleo de la lista de ingredientes en un único vector denso,
      evitando mezclar pasos distintos dentro de un mismo embedding y maximizando
      la precisión coseno del modelo `bge-m3`.
    - chunk_overlap = 50 caracteres:
      Un solapamiento del 12.5% (50 caracteres) preserva la continuidad sintáctica
      y léxica entre fragmentos contiguos (por ejemplo, cuando una lista de
      ingredientes larga o un paso con tiempos de cocción se extiende al límite
      del fragmento), evitando la pérdida de ingredientes o conectores clave en las
      fronteras de corte sin inflar innecesariamente la cardinalidad de ChromaDB.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Rutas y constantes de configuración
BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "recetario_paulina_cocina"
EMBEDDING_MODEL = "bge-m3"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def extraer_metadatos_encabezado(texto: str, nombre_archivo: str) -> dict[str, str]:
    """
    Analiza las primeras líneas del archivo .txt para extraer los metadatos
    declarados en el encabezado (Título, URL de origen, Fecha de captura, Autor, Licencia).
    """
    metadatos: dict[str, str] = {
        "archivo": nombre_archivo,
        "receta": nombre_archivo.replace(".txt", ""),
        "url_origen": "https://www.paulinacocina.net",
        "fecha_captura": "Desconocida",
        "autor": "Paulina Cocina",
        "licencia": "Uso didáctico",
    }

    patrones = {
        "receta": r"^T[íi]tulo:\s*(.+)$",
        "url_origen": r"^URL de origen:\s*(.+)$",
        "fecha_captura": r"^Fecha de captura:\s*(.+)$",
        "autor": r"^Autor:\s*(.+)$",
        "licencia": r"^Licencia/Condici[óo]n de uso:\s*(.+)$",
    }

    for linea in texto.splitlines()[:10]:
        for clave, patron in patrones.items():
            coincidencia = re.match(patron, linea.strip(), flags=re.IGNORECASE)
            if coincidencia:
                metadatos[clave] = coincidencia.group(1).strip()

    return metadatos


def cargar_documentos(directorio_datos: Path = DATOS_DIR) -> list[Document]:
    """
    Carga todos los archivos `.txt` de la carpeta `datos/` utilizando `TextLoader`
    e inyecta los metadatos trazables extraídos de sus encabezados.
    """
    if not directorio_datos.exists():
        raise FileNotFoundError(
            f"No existe el directorio de datos en '{directorio_datos}'. "
            "Ejecutá primero `python scraper.py`."
        )

    archivos_txt = sorted(directorio_datos.glob("*.txt"))
    if not archivos_txt:
        raise FileNotFoundError(
            f"No se encontraron archivos .txt dentro de '{directorio_datos}'."
        )

    documentos: list[Document] = []
    for ruta_txt in archivos_txt:
        loader = TextLoader(str(ruta_txt), encoding="utf-8")
        docs_archivo = loader.load()
        for doc in docs_archivo:
            meta_extraida = extraer_metadatos_encabezado(
                doc.page_content, ruta_txt.name
            )
            doc.metadata.update(meta_extraida)
            documentos.append(doc)
            print(
                f"  [+] Cargado: {ruta_txt.name:<32} | "
                f"Receta: '{meta_extraida['receta']}' ({len(doc.page_content)} caracteres)"
            )

    return documentos


def segmentar_documentos(
    documentos: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Divide los documentos en fragmentos usando `RecursiveCharacterTextSplitter`
    con `chunk_size=400` y `chunk_overlap=50`, preservando y enriqueciendo los
    metadatos de cada chunk con su índice y el nombre explícito de la receta.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    fragmentos = splitter.split_documents(documentos)

    # Numerar cada fragmento y asegurar que el título de la receta sea visible
    # dentro del texto del chunk para evitar ambigüedad semántica entre recetas
    conteo_por_archivo: dict[str, int] = {}
    for chunk in fragmentos:
        archivo = chunk.metadata.get("archivo", "desconocido.txt")
        receta = chunk.metadata.get("receta", "Receta")
        conteo_por_archivo[archivo] = conteo_por_archivo.get(archivo, 0) + 1
        chunk.metadata["chunk_id"] = conteo_por_archivo[archivo]

        # Si el fragmento no menciona el nombre de la receta en su texto, se antepone
        # un encabezado breve para reforzar la correspondencia ingrediente <-> receta
        prefijo = f"[Receta: {receta}]\n"
        if receta.lower() not in chunk.page_content.lower():
            chunk.page_content = prefijo + chunk.page_content

    return fragmentos


def construir_indice_vectorial(
    fragmentos: list[Document],
    directorio_chroma: Path = CHROMA_DIR,
    limpiar_existente: bool = True,
) -> Chroma:
    """
    Genera los embeddings con `OllamaEmbeddings(model="bge-m3")` y persiste la
    colección vectorial en `./chroma_db` mediante `Chroma.from_documents`.
    """
    if limpiar_existente and directorio_chroma.exists():
        print(f"[*] Eliminando índice previo en '{directorio_chroma}' para reindexación limpia...")
        shutil.rmtree(directorio_chroma)

    print(f"[*] Inicializando modelo de embeddings local: OllamaEmbeddings(model='{EMBEDDING_MODEL}')...")
    # num_gpu=0 fuerza inferencia en CPU si el driver CUDA local tiene incompatibilidad PTX con Ollama
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, num_gpu=0)

    print(
        f"[*] Indexando {len(fragmentos)} fragmentos en ChromaDB "
        f"(directorio: '{directorio_chroma}', colección: '{COLLECTION_NAME}')..."
    )
    vectorstore = Chroma.from_documents(
        documents=fragmentos,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(directorio_chroma),
    )

    return vectorstore


def ejecutar_ingesta() -> None:
    """Flujo principal reproducible de ingesta e indexación desde línea de comandos."""
    print("=" * 75)
    print("INGESTA E INDEXACIÓN VECTORIAL RAG - PAULINA COCINA (IFTS N.º 24)")
    print("=" * 75)

    try:
        print("\n1. Cargando documentos desde 'datos/' con TextLoader...")
        documentos = cargar_documentos(DATOS_DIR)

        print(
            f"\n2. Segmentando documentos con RecursiveCharacterTextSplitter "
            f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})..."
        )
        fragmentos = segmentar_documentos(documentos)
        print(f"   -> Total de fragmentos generados: {len(fragmentos)}")

        # Mostrar resumen de fragmentos por documento
        resumen: dict[str, int] = {}
        for frag in fragmentos:
            arch = frag.metadata.get("archivo", "desconocido")
            resumen[arch] = resumen.get(arch, 0) + 1
        for arch, cant in resumen.items():
            print(f"      - {arch}: {cant} chunks")

        print("\n3. Creando base vectorial persistente en './chroma_db'...")
        construir_indice_vectorial(
            fragmentos=fragmentos,
            directorio_chroma=CHROMA_DIR,
            limpiar_existente=True,
        )

        print("\n" + "=" * 75)
        print("[ÉXITO] Reindexación completada correctamente en './chroma_db'.")
        print("=" * 75)

    except Exception as exc:
        print(f"\n[ERROR CRÍTICO EN INGESTA]: {exc}", file=sys.stderr)
        print(
            "Verificá que el servidor Ollama esté corriendo (`ollama serve`) y que "
            f"hayas descargado el modelo de embeddings con: `ollama pull {EMBEDDING_MODEL}`",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    ejecutar_ingesta()
