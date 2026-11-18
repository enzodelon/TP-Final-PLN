"""
scraper.py - Descargador reproducible de recetas desde Paulina Cocina (www.paulinacocina.net).

Descripción:
    Descarga o actualiza el corpus de recetas desde www.paulinacocina.net utilizando
    requests y BeautifulSoup4. Extrae datos estructurados (JSON-LD Schema.org Recipe
    cuando está disponible y nodos HTML del artículo como respaldo), limpia el texto
    y escribe cada receta en la carpeta `datos/` con su encabezado de metadatos
    obligatorio (Título, URL de origen, Fecha de captura, Autor y Licencia).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup

# Directorio destino del corpus
DATOS_DIR = Path(__file__).resolve().parent / "datos"

# Catálogo de recetas objetivo del corpus
RECETAS_OBJETIVO: list[dict[str, str]] = [
    {
        "archivo": "receta_01_guiso_lentejas.txt",
        "titulo_fallback": "Guiso de lentejas fácil y tradicional",
        "url": "https://www.paulinacocina.net/guiso-de-lentejas-receta-facil/10644",
    },
    {
        "archivo": "receta_02_tortilla_papas.txt",
        "titulo_fallback": "Tortilla de papas española perfecta (jugosa o cocida)",
        "url": "https://www.paulinacocina.net/tortilla-de-papas-espanola/10599",
    },
    {
        "archivo": "receta_03_empanadas_carne.txt",
        "titulo_fallback": "Empanadas de carne cortada a cuchillo o picada (bien jugosas)",
        "url": "https://www.paulinacocina.net/como-hacer-empanadas-de-carne/7225",
    },
]

HEADERS_HTTP: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "BotAcademico-IFTS24-RAG/1.0 (+https://www.paulinacocina.net)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-AR,es;q=0.9,en;q=0.5",
}

LICENCIA_USO = (
    "Uso exclusivamente didáctico y académico (Trabajo Integrador 2 - IFTS N.º 24)"
)


def limpiar_espacios(texto: str) -> str:
    """Normaliza espacios en blanco, saltos de línea múltiples y entidades residuales."""
    texto = re.sub(r"\r\n|\r", "\n", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _buscar_nodo_receta_jsonld(data: Any) -> dict[str, Any] | None:
    """Recorre recursivamente una estructura JSON-LD buscando un nodo `@type: Recipe`."""
    if isinstance(data, dict):
        tipo = data.get("@type")
        if tipo == "Recipe" or (isinstance(tipo, list) and "Recipe" in tipo):
            return data
        if "@graph" in data and isinstance(data["@graph"], list):
            for item in data["@graph"]:
                encontrado = _buscar_nodo_receta_jsonld(item)
                if encontrado:
                    return encontrado
    elif isinstance(data, list):
        for item in data:
            encontrado = _buscar_nodo_receta_jsonld(item)
            if encontrado:
                return encontrado
    return None


def extraer_desde_jsonld(soup: BeautifulSoup) -> dict[str, Any] | None:
    """Intenta extraer título, descripción, ingredientes y pasos desde Schema.org Recipe."""
    for script in soup.find_all("script", type="application/ld+json"):
        contenido = script.string
        if not contenido:
            continue
        try:
            data = json.loads(contenido)
        except json.JSONDecodeError:
            continue

        nodo = _buscar_nodo_receta_jsonld(data)
        if not nodo:
            continue

        titulo = limpiar_espacios(str(nodo.get("name", "")))
        descripcion = limpiar_espacios(str(nodo.get("description", "")))

        ingredientes_raw = nodo.get("recipeIngredient", [])
        ingredientes: list[str] = [
            limpiar_espacios(BeautifulSoup(str(ing), "html.parser").get_text(" "))
            for ing in ingredientes_raw
            if str(ing).strip()
        ]

        instrucciones_raw = nodo.get("recipeInstructions", [])
        pasos: list[str] = []
        if isinstance(instrucciones_raw, list):
            for idx, inst in enumerate(instrucciones_raw, start=1):
                if isinstance(inst, dict):
                    texto_paso = inst.get("text") or inst.get("name") or ""
                    texto_limpio = limpiar_espacios(
                        BeautifulSoup(str(texto_paso), "html.parser").get_text(" ")
                    )
                    if texto_limpio:
                        pasos.append(f"Paso {idx}: {texto_limpio}")
                elif isinstance(inst, str) and inst.strip():
                    pasos.append(f"Paso {idx}: {limpiar_espacios(inst)}")
        elif isinstance(instrucciones_raw, str) and instrucciones_raw.strip():
            pasos.append(limpiar_espacios(instrucciones_raw))

        if ingredientes and pasos:
            return {
                "titulo": titulo,
                "descripcion": descripcion,
                "ingredientes": ingredientes,
                "pasos": pasos,
            }
    return None


def extraer_desde_html(soup: BeautifulSoup, titulo_fallback: str) -> dict[str, Any]:
    """Extrae título, ingredientes y preparación directamente del DOM HTML del artículo."""
    h1 = soup.find("h1")
    titulo = limpiar_espacios(h1.get_text(" ")) if h1 else titulo_fallback

    # Eliminar etiquetas ruidosas antes de procesar el cuerpo
    for tag in soup(["script", "style", "nav", "footer", "aside", "form", "iframe"]):
        tag.decompose()

    contenedor = (
        soup.find("article")
        or soup.find("div", class_=re.compile(r"entry-content|post-content|wprm-recipe"))
        or soup.body
    )

    ingredientes: list[str] = []
    pasos: list[str] = []
    parrafos_descriptivos: list[str] = []

    if contenedor:
        # Intento específico con clases de WP Recipe Maker / listas de ingredientes
        nodos_ing = contenedor.select(
            ".wprm-recipe-ingredient, .ingredients li, ul li"
        )
        for li in nodos_ing:
            txt = limpiar_espacios(li.get_text(" "))
            if 3 < len(txt) < 180:
                ingredientes.append(txt)

        nodos_pasos = contenedor.select(
            ".wprm-recipe-instruction, .instructions li, ol li"
        )
        for idx, li in enumerate(nodos_pasos, start=1):
            txt = limpiar_espacios(li.get_text(" "))
            if len(txt) > 15:
                pasos.append(f"Paso {idx}: {txt}")

        for p in contenedor.find_all("p"):
            txt = limpiar_espacios(p.get_text(" "))
            if len(txt) > 40:
                parrafos_descriptivos.append(txt)

    descripcion = (
        "\n".join(parrafos_descriptivos[:2])
        if parrafos_descriptivos
        else f"Receta tradicional de {titulo} extraída de Paulina Cocina."
    )

    # Si no se detectaron pasos en <ol>, usar los párrafos del artículo
    if not pasos and parrafos_descriptivos:
        for idx, p_txt in enumerate(parrafos_descriptivos[2:10], start=1):
            pasos.append(f"Paso {idx}: {p_txt}")

    return {
        "titulo": titulo,
        "descripcion": descripcion,
        "ingredientes": ingredientes[:25],
        "pasos": pasos,
    }


def formatear_documento_receta(
    titulo: str,
    url: str,
    fecha_captura: str,
    descripcion: str,
    ingredientes: list[str],
    pasos: list[str],
) -> str:
    """Construye el archivo de texto plano con encabezados de metadatos trazables."""
    lineas: list[str] = [
        f"Título: {titulo}",
        f"URL de origen: {url}",
        f"Fecha de captura: {fecha_captura}",
        "Autor: Paulina Cocina",
        f"Licencia/Condición de uso: {LICENCIA_USO}",
        "",
        "=== DESCRIPCIÓN DE LA RECETA ===",
        descripcion,
        "",
        f"=== INGREDIENTES ({titulo}) ===",
    ]
    for ing in ingredientes:
        prefijo = "" if ing.startswith("-") else "- "
        lineas.append(f"{prefijo}{ing}")

    lineas.append("")
    lineas.append(f"=== INSTRUCCIONES PASO A PASO ({titulo}) ===")
    for paso in pasos:
        lineas.append(paso)
        lineas.append("")

    return "\n".join(lineas).strip() + "\n"


def descargar_receta(
    item: dict[str, str],
    pausa_segundos: float = 2.0,
    sobrescribir: bool = False,
) -> bool:
    """
    Descarga una receta desde su URL, extrae su contenido y la guarda en `datos/`.
    Si el archivo ya existe y `sobrescribir=False`, respeta la copia local verificada.
    """
    DATOS_DIR.mkdir(parents=True, exist_ok=True)
    ruta_destino = DATOS_DIR / item["archivo"]

    if ruta_destino.exists() and not sobrescribir:
        print(
            f"[OK] '{item['archivo']}' ya existe en '{DATOS_DIR}'. "
            "Usá --sobrescribir para volver a descargarlo desde la web."
        )
        return True

    url = item["url"]
    print(f"[SCRAPER] Consultando: {url} ...")
    try:
        respuesta = requests.get(url, headers=HEADERS_HTTP, timeout=20)
        respuesta.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ADVERTENCIA] No se pudo conectar a {url}: {exc}")
        if ruta_destino.exists():
            print(f"              Se conserva el archivo local existente: {ruta_destino.name}")
            return True
        return False

    soup = BeautifulSoup(respuesta.text, "html.parser")
    datos_extraidos = extraer_desde_jsonld(soup)
    if not datos_extraidos:
        datos_extraidos = extraer_desde_html(soup, item["titulo_fallback"])

    if not datos_extraidos["ingredientes"] or not datos_extraidos["pasos"]:
        print(
            f"[ADVERTENCIA] Extracción incompleta en {url}. "
            "Se conserva el archivo curado si ya existe."
        )
        return ruta_destino.exists()

    fecha_hoy = datetime.date.today().isoformat()
    contenido_txt = formatear_documento_receta(
        titulo=datos_extraidos["titulo"] or item["titulo_fallback"],
        url=url,
        fecha_captura=fecha_hoy,
        descripcion=datos_extraidos["descripcion"],
        ingredientes=datos_extraidos["ingredientes"],
        pasos=datos_extraidos["pasos"],
    )

    ruta_destino.write_text(contenido_txt, encoding="utf-8")
    print(f"[GUARDADO] {ruta_destino} ({len(contenido_txt)} caracteres)")

    if pausa_segundos > 0:
        time.sleep(pausa_segundos)
    return True


def ejecutar_scraping(sobrescribir: bool = False, pausa_segundos: float = 2.0) -> None:
    """Recorre todas las recetas objetivo y verifica/actualiza la carpeta `datos/`."""
    print("=" * 70)
    print("SCRAPER DE RECETAS - PAULINA COCINA (Corpus RAG - IFTS N.º 24)")
    print("=" * 70)

    exitos = 0
    for item in RECETAS_OBJETIVO:
        if descargar_receta(
            item=item,
            pausa_segundos=pausa_segundos,
            sobrescribir=sobrescribir,
        ):
            exitos += 1

    print("-" * 70)
    print(
        f"Proceso finalizado: {exitos}/{len(RECETAS_OBJETIVO)} recetas listas en '{DATOS_DIR}'."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga o verifica el corpus de recetas de Paulina Cocina en datos/."
    )
    parser.add_argument(
        "--sobrescribir",
        action="store_true",
        help="Fuerza la descarga HTTP y sobrescribe los archivos locales en datos/.",
    )
    parser.add_argument(
        "--pausa",
        type=float,
        default=2.0,
        help="Segundos de pausa de cortesía entre peticiones HTTP (por defecto: 2.0).",
    )
    args = parser.parse_args()
    ejecutar_scraping(sobrescribir=args.sobrescribir, pausa_segundos=args.pausa)
