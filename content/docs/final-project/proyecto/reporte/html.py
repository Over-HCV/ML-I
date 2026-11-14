"""Capa 7 · Componentes HTML para presentar resultados.

Responsabilidad única: convertir resultados ya calculados en piezas visuales
con aspecto de diapositiva (encabezados, tarjetas, tablas, barras, embudo,
línea de tiempo, pestañas). Cada función devuelve un objeto
`IPython.display.HTML`; quien llama lo pasa a `salida.mostrar`.

Criterios de diseño:

- Cada pieza lleva su propia hoja de estilos, porque Colab dibuja cada salida
  en un marco aislado. Las clases usan el prefijo `pf-`.
- Todo el contenido y el movimiento esencial funcionan solo con CSS. El
  guion final es un adorno (conteo de cifras y espera a que la pieza entre en
  pantalla): si el visor no lo ejecuta, el resultado es igualmente correcto.
- Fondo y colores de texto propios, legibles en temas claros y oscuros.
- Se respeta `prefers-reduced-motion`.
- El texto recibido se escapa siempre. Solo se interpretan dos marcas
  mínimas: `**negrita**` y `` `código` ``.
"""
import base64
import hashlib
import io
import math
import re
from html import escape

from IPython.display import HTML

from proyecto.base import config
from proyecto.base.config import COLORES, ETAPAS, TITULOS_ETAPAS

ROTULO_PROYECTO = getattr(config, "ROTULO_PROYECTO",
                          "Proyecto final · Aprendizaje Automático de Máquina I")
TITULO_PROYECTO = getattr(config, "TITULO_PROYECTO",
                          "Aprendizaje automático sobre la base de muestras de un "
                          "laboratorio de microbiología")

TENUE = "#52606D"      # texto secundario (contraste suficiente sobre blanco)
LINEA = "#D9DEE5"      # bordes y pistas
TIPOS_NOTA = {
    "info": ("azul", "Nota"),
    "hallazgo": ("verde", "Hallazgo"),
    "advertencia": ("naranja", "Advertencia"),
    "limite": ("rojo", "Límite"),
    "simulacion": ("morado", "Datos simulados"),
}
MAXIMO_PESTANAS = 8
_contador = [0]


# ----------------------------------------------------------------------
# Hojas de estilo (una base común y una por componente)
# ----------------------------------------------------------------------
_CURVA = "cubic-bezier(.2,.7,.2,1)"
_CSS = {}
_CSS["base"] = f"""
.pf{{--pf-azul:{COLORES['azul']};--pf-rojo:{COLORES['rojo']};--pf-verde:{COLORES['verde']};
--pf-naranja:{COLORES['naranja']};--pf-morado:{COLORES['morado']};--pf-gris:{COLORES['gris']};
--pf-tinta:{COLORES['tinta']};--pf-fondo:{COLORES['fondo']};--pf-tenue:{TENUE};--pf-linea:{LINEA};
box-sizing:border-box;position:relative;max-width:980px;margin:10px 0;padding:20px 24px;
background:#fff;color:var(--pf-tinta);border:1px solid var(--pf-linea);border-radius:10px;
font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
font-size:15px;line-height:1.45;text-align:left}}
.pf *{{box-sizing:border-box}}
.pf .pf:not(.pf-nota):not(.pf-encabezado){{border:0;padding:0;margin:0;background:transparent;max-width:none}}
.pf .pf-titulo{{font-size:17px;font-weight:650;margin:0 0 14px;color:var(--pf-tinta)}}
.pf code{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em;
background:#EEF1F5;color:var(--pf-tinta);padding:1px 5px;border-radius:4px}}
.pf strong{{font-weight:650;color:inherit}}
@keyframes pf-aparecer{{from{{opacity:0;transform:translateY(10px)}}}}
@keyframes pf-crecer{{from{{transform:scaleX(0)}}}}
@keyframes pf-abrir{{from{{clip-path:inset(0 50%);opacity:.4}}to{{clip-path:inset(0 0)}}}}
@keyframes pf-pulso{{50%{{box-shadow:0 0 0 8px rgba(76,114,176,.12)}}}}
.pf .pf-ap{{animation:pf-aparecer .55s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*90ms)}}
.pf.pf-pausa,.pf.pf-pausa *{{animation-play-state:paused!important}}
@media (prefers-reduced-motion:reduce){{.pf,.pf *{{animation:none!important;transition:none!important}}}}
"""
_CSS["encabezado"] = """
.pf.pf-encabezado{background:var(--pf-tinta);border-color:var(--pf-tinta);color:#fff;padding:26px 28px 22px}
.pf-enc-rotulo{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#B8C4D0}
.pf-enc-proyecto{font-size:14px;color:#D5DCE4;margin-top:2px}
.pf-enc-titulo{font-size:30px;font-weight:700;line-height:1.15;margin:14px 0 4px;color:#fff}
.pf-enc-sub{font-size:15px;color:#D5DCE4;max-width:72ch}
.pf-progreso{display:flex;margin-top:24px}
.pf-prog-paso{flex:1 1 0;min-width:0;position:relative;padding:22px 4px 0;text-align:center;
font-size:11.5px;line-height:1.25;color:#8FA0B2}
.pf-prog-paso::before{content:"";position:absolute;top:6px;left:-50%;width:100%;height:2px;background:#46525F}
.pf-prog-paso:first-child::before{display:none}
.pf-prog-punto{position:absolute;top:0;left:50%;width:14px;height:14px;margin-left:-7px;
border-radius:50%;background:var(--pf-tinta);border:2px solid #5B6875;z-index:1}
.pf-prog-hecho{color:#C9D3DD}
.pf-prog-hecho::before,.pf-prog-actual::before{background:var(--pf-verde)}
.pf-prog-hecho .pf-prog-punto{background:var(--pf-verde);border-color:var(--pf-verde)}
.pf-prog-actual{color:#fff;font-weight:650}
.pf-prog-actual .pf-prog-punto{background:#fff;border-color:var(--pf-azul);
box-shadow:0 0 0 4px rgba(76,114,176,.5);animation:pf-pulso 2.4s ease-in-out 3}
"""
_CSS["tarjetas"] = """
.pf-tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}
.pf-tarjeta{background:var(--pf-fondo);border-radius:8px;padding:16px 18px;border-top:3px solid var(--pf-c,var(--pf-azul))}
.pf-tar-titulo{font-size:12.5px;color:var(--pf-tenue);text-transform:uppercase;letter-spacing:.05em}
.pf-tar-valor{font-size:32px;font-weight:700;line-height:1.15;margin:6px 0 2px;color:var(--pf-tinta);
font-variant-numeric:tabular-nums}
.pf-tar-detalle{font-size:13px;color:var(--pf-tenue)}
"""
_CSS["nota"] = """
.pf.pf-nota{border-left:5px solid var(--pf-c,var(--pf-azul));padding:14px 20px;background:#fff}
.pf-nota-titulo{font-size:12.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
color:var(--pf-tinta);margin-bottom:4px}
.pf-nota-titulo::before{content:"";display:inline-block;width:9px;height:9px;border-radius:50%;
background:var(--pf-c,var(--pf-azul));margin-right:8px}
.pf-nota-texto{color:var(--pf-tinta);max-width:80ch}
"""
_CSS["tabla"] = f"""
.pf-tabla-marco{{overflow-x:auto}}
.pf .pf-tabla{{border-collapse:collapse;width:auto;min-width:min(100%,460px);margin:0;border:0;font-size:14px;
color:var(--pf-tinta);background:#fff;table-layout:auto}}
.pf .pf-tabla caption.pf-tabla-titulo{{caption-side:top;text-align:left;font-size:15px;font-weight:650;
color:var(--pf-tinta);padding:0 0 10px}}
.pf .pf-tabla thead tr.pf-fila-cab{{background:#fff}}
.pf .pf-tabla thead th.pf-c{{padding:8px 12px;text-align:left;font-weight:650;color:var(--pf-tinta);
background:#fff;border:0;border-bottom:2px solid var(--pf-tinta);white-space:nowrap}}
.pf .pf-tabla tbody tr.pf-fila{{background:#fff}}
.pf .pf-tabla tbody tr.pf-fila:hover{{background:var(--pf-fondo)}}
.pf .pf-tabla tbody td.pf-c{{padding:7px 12px;text-align:left;vertical-align:middle;color:var(--pf-tinta);
border:0;border-bottom:1px solid var(--pf-linea)}}
.pf .pf-tabla thead th.pf-num,.pf .pf-tabla tbody td.pf-num{{text-align:right;font-variant-numeric:tabular-nums}}
.pf .pf-tabla thead th.pf-col-res,.pf .pf-tabla tbody td.pf-col-res{{background:#EEF2F8}}
.pf .pf-tabla tbody tr.pf-fila-res,.pf .pf-tabla tbody tr.pf-fila-res:hover{{background:#E3EAF5}}
.pf .pf-tabla tbody tr.pf-fila-res td.pf-c{{font-weight:650}}
.pf .pf-tabla tbody tr.pf-fila-res td.pf-c:first-child{{box-shadow:inset 4px 0 0 var(--pf-azul)}}
.pf-celda-barra{{display:flex;align-items:center;justify-content:flex-end;gap:10px}}
.pf-celda-pista{{flex:1 1 70px;min-width:70px;max-width:180px;height:8px;background:#E9EDF2;border-radius:4px;overflow:hidden}}
.pf-celda-relleno{{display:block;height:100%;width:var(--pf-w,0%);background:var(--pf-c,var(--pf-azul));
border-radius:4px;transform-origin:left;animation:pf-crecer .8s {_CURVA} both;
animation-delay:calc(var(--pf-i,0)*60ms)}}
.pf-tabla-pie{{font-size:12.5px;color:var(--pf-tenue);margin-top:10px}}
"""
_CSS["barras"] = f"""
.pf-barras{{display:grid;grid-template-columns:fit-content(38%) 1fr max-content;gap:10px 12px;align-items:center}}
.pf-bar-etq{{font-size:14px;text-align:right;color:var(--pf-tinta)}}
.pf-bar-pista{{height:18px;background:#EEF1F5;border-radius:4px;overflow:hidden}}
.pf-bar-relleno{{height:100%;width:var(--pf-w,0%);background:var(--pf-c,var(--pf-azul));border-radius:4px;
transform-origin:left;animation:pf-crecer .9s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*80ms)}}
.pf-bar-valor{{font-size:14px;font-weight:650;font-variant-numeric:tabular-nums;color:var(--pf-tinta)}}
.pf-bar-res{{font-weight:700}}
"""
_CSS["embudo"] = f"""
.pf-embudo{{display:flex;flex-direction:column;align-items:center}}
.pf-emb-nivel{{width:var(--pf-w,100%);min-width:230px;max-width:100%;background:var(--pf-c,var(--pf-azul));
color:#fff;border-radius:6px;padding:12px 16px;text-align:center;
animation:pf-abrir .75s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*260ms)}}
.pf-emb-valor{{font-size:24px;font-weight:700;line-height:1.15;color:#fff;font-variant-numeric:tabular-nums}}
.pf-emb-etq{{font-size:13.5px;color:#fff}}
.pf-emb-paso{{font-size:12.5px;color:var(--pf-tenue);padding:6px 0;text-align:center;max-width:60ch;
animation:pf-aparecer .5s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*260ms + 130ms)}}
.pf-emb-paso::before{{content:"\\25BE";display:block;font-size:16px;line-height:1;color:var(--pf-gris)}}
"""
_CSS["linea"] = """
.pf-linea-t{display:flex}
.pf-hito{flex:1 1 0;min-width:0;position:relative;padding:28px 8px 0;text-align:center}
.pf-hito::before{content:"";position:absolute;top:7px;left:0;right:0;height:2px;background:var(--pf-linea)}
.pf-hito:first-child::before{left:50%}
.pf-hito:last-child::before{right:50%}
.pf-hito-punto{position:absolute;top:0;left:50%;margin-left:-8px;width:16px;height:16px;border-radius:50%;
background:var(--pf-c,var(--pf-azul));border:3px solid #fff;box-shadow:0 0 0 1px var(--pf-c,var(--pf-azul))}
.pf-hito-fecha{font-size:12.5px;color:var(--pf-tenue);font-variant-numeric:tabular-nums}
.pf-hito-titulo{font-size:14.5px;font-weight:650;color:var(--pf-tinta)}
.pf-hito-det{font-size:13px;color:var(--pf-tenue)}
"""
_CSS["pasos"] = """
.pf-paso{display:grid;grid-template-columns:34px 1fr;column-gap:14px;position:relative;padding-bottom:16px}
.pf-paso:last-child{padding-bottom:0}
.pf-paso::before{content:"";position:absolute;left:16px;top:36px;bottom:2px;width:2px;background:var(--pf-linea)}
.pf-paso:last-child::before{display:none}
.pf-paso-num{width:34px;height:34px;border-radius:50%;background:var(--pf-azul);color:#fff;font-size:14px;
font-weight:700;display:flex;align-items:center;justify-content:center}
.pf-paso-titulo{font-weight:650;padding-top:6px;color:var(--pf-tinta)}
.pf-paso-det{font-size:14px;color:var(--pf-tenue);max-width:78ch}
.pf-paso-pend .pf-paso-num{background:#fff;color:var(--pf-tenue);border:2px solid var(--pf-linea)}
.pf-paso-pend .pf-paso-titulo{color:var(--pf-tenue)}
.pf-paso-act .pf-paso-num{box-shadow:0 0 0 4px rgba(76,114,176,.25)}
"""
_CSS["pestanas"] = """
.pf-pestanas>input.pf-radio{position:absolute;opacity:0;pointer-events:none}
.pf-lenguetas{display:flex;flex-wrap:wrap;gap:4px;border-bottom:2px solid var(--pf-linea);margin-bottom:16px}
.pf-lenguetas label{padding:8px 14px;margin-bottom:-2px;font-size:14px;font-weight:600;color:var(--pf-tenue);
cursor:pointer;border-bottom:3px solid transparent;border-radius:6px 6px 0 0;
transition:color .2s,border-color .2s,background .2s}
.pf-lenguetas label:hover{background:var(--pf-fondo);color:var(--pf-tinta)}
.pf-paneles>.pf-panel{display:none}
.pf-panel img{max-width:100%;height:auto;display:block}
""" + "".join(
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):checked~.pf-lenguetas label:nth-of-type({k})"
    "{color:var(--pf-azul);border-bottom-color:var(--pf-azul)}\n"
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):focus-visible~.pf-lenguetas label:nth-of-type({k})"
    "{outline:2px solid var(--pf-azul);outline-offset:2px}\n"
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):checked~.pf-paneles>.pf-panel:nth-of-type({k})"
    "{display:block;animation:pf-aparecer .4s ease-out both}\n"
    for k in range(1, MAXIMO_PESTANAS + 1)
)
_CSS["comparacion"] = """
.pf-comparacion{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;padding-top:6px}
.pf-opcion{position:relative;border:1px solid var(--pf-linea);border-radius:8px;padding:16px 18px;background:#fff}
.pf-opcion.pf-op-elegida{border:2px solid var(--pf-azul);background:#F3F6FB}
.pf-op-sello{position:absolute;top:-11px;right:14px;background:var(--pf-azul);color:#fff;font-size:11.5px;
font-weight:650;letter-spacing:.05em;text-transform:uppercase;padding:2px 10px;border-radius:10px}
.pf-op-titulo{font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--pf-tenue)}
.pf-op-valor{font-size:28px;font-weight:700;line-height:1.2;margin-top:4px;color:var(--pf-tinta);
font-variant-numeric:tabular-nums}
.pf-op-etq{font-size:13px;color:var(--pf-tenue)}
.pf-op-puntos{margin-top:10px;font-size:14px;color:var(--pf-tinta)}
.pf-op-punto{position:relative;padding-left:14px;margin:4px 0}
.pf-op-punto::before{content:"";position:absolute;left:0;top:.6em;width:5px;height:5px;border-radius:50%;
background:var(--pf-gris)}
"""

# Adorno opcional: espera a que la pieza entre en pantalla para animarla y
# cuenta las cifras desde cero. El texto final ya está en el HTML.
_GUION = """<script>(function(){try{
var quieto=window.matchMedia&&window.matchMedia('(prefers-reduced-motion: reduce)').matches;
var piezas=document.querySelectorAll('.pf-raiz:not([data-pf-listo])');
Array.prototype.forEach.call(piezas,function(pieza){
pieza.setAttribute('data-pf-listo','1');if(quieto)return;
function contar(){Array.prototype.forEach.call(pieza.querySelectorAll('[data-pf-valor]'),function(cifra){
var fin=cifra.textContent,valor=parseFloat(cifra.getAttribute('data-pf-valor')),
dec=parseInt(cifra.getAttribute('data-pf-dec')||'0',10),pre=cifra.getAttribute('data-pf-pre')||'',
suf=cifra.getAttribute('data-pf-suf')||'',inicio=null;
if(!isFinite(valor)||!window.requestAnimationFrame)return;
function paso(t){if(inicio===null)inicio=t;var p=Math.min((t-inicio)/900,1),q=1-Math.pow(1-p,3);
cifra.textContent=p<1?pre+(valor*q).toFixed(dec)+suf:fin;if(p<1)requestAnimationFrame(paso);}
requestAnimationFrame(paso);setTimeout(function(){cifra.textContent=fin;},2500);});}
if('IntersectionObserver' in window){pieza.classList.add('pf-pausa');
var ojo=new IntersectionObserver(function(e){for(var i=0;i<e.length;i++){if(e[i].isIntersecting){
ojo.disconnect();pieza.classList.remove('pf-pausa');contar();break;}}},{threshold:0});
ojo.observe(pieza);setTimeout(function(){pieza.classList.remove('pf-pausa');},60000);}else{contar();}
});}catch(e){Array.prototype.forEach.call(document.querySelectorAll('.pf-pausa'),
function(p){p.classList.remove('pf-pausa');});}})();</script>"""


# ----------------------------------------------------------------------
# Utilidades internas
# ----------------------------------------------------------------------
def _texto(valor):
    """Texto escapado, con `**negrita**` y `` `código` `` como únicas marcas."""
    texto = escape(str(valor))
    texto = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", texto)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", texto)


def _color(nombre, defecto="azul"):
    """Color de `config.COLORES` por nombre, o un hexadecimal válido."""
    if nombre in COLORES:
        return COLORES[nombre]
    if isinstance(nombre, str) and re.fullmatch(r"#[0-9A-Fa-f]{3,8}", nombre):
        return nombre
    return COLORES[defecto]


def _mezclar(color_a, color_b, proporcion):
    """Color intermedio entre dos hexadecimales `#RRGGBB`."""
    a = [int(color_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(color_b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * proporcion):02X}" for x, y in zip(a, b))


def _es_numero(valor):
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) \
        or (hasattr(valor, "dtype") and getattr(valor, "shape", None) == ()
            and valor.dtype.kind in "iuf")


def _falta(valor):
    if valor is None:
        return True
    try:
        return bool(valor != valor)   # NaN y NaT no son iguales a sí mismos
    except (TypeError, ValueError):
        return False


def formato_numero(valor, decimales=4):
    """Formato único del proyecto: punto decimal, sin separador de miles,
    enteros sin decimales, «—» para los faltantes.
    Uso: `formato_numero(0.81234, 3)` → `'0.812'`."""
    if _falta(valor):
        return "—"
    if isinstance(valor, bool) or (hasattr(valor, "dtype") and valor.dtype.kind == "b"):
        return "sí" if valor else "no"
    if _es_numero(valor):
        if isinstance(valor, int) or (hasattr(valor, "dtype") and valor.dtype.kind in "iu"):
            return str(int(valor))
        if math.isinf(float(valor)):
            return "∞" if valor > 0 else "−∞"
        return f"{float(valor):.{int(decimales)}f}"
    if hasattr(valor, "strftime"):
        con_hora = getattr(valor, "hour", 0) or getattr(valor, "minute", 0)
        return valor.strftime("%Y-%m-%d %H:%M" if con_hora else "%Y-%m-%d")
    return str(valor)


def _aplicar_formato(valor, formato):
    """`formato` puede ser un número de decimales, una plantilla de
    `str.format` (por ejemplo `"{:.1f} %"`) o una función."""
    if callable(formato):
        return str(formato(valor))
    if isinstance(formato, str):
        return "—" if _falta(valor) else formato.format(valor)
    return formato_numero(valor, formato)


def _pares(elementos):
    """Acepta un diccionario, una Serie de pandas o una lista de pares."""
    if hasattr(elementos, "items"):
        return list(elementos.items())
    return [tuple(e) for e in elementos]


def _campos(elemento, nombres):
    """Normaliza un elemento dado como diccionario, tupla o texto suelto."""
    if isinstance(elemento, dict):
        return dict(elemento)
    if isinstance(elemento, (str, bytes)) or not hasattr(elemento, "__iter__"):
        elemento = (elemento,)
    return dict(zip(nombres, elemento))


def _envolver(cuerpo, componente, clases="", estilo="", titulo=None):
    """Raíz de un componente: estilos propios + contenido + adorno opcional."""
    hoja = _CSS["base"] + _CSS.get(componente, "")
    cabecera = f'<div class="pf-titulo">{_texto(titulo)}</div>' if titulo else ""
    atributo = f' style="{estilo}"' if estilo else ""
    return HTML(
        f"<style>{hoja}</style>"
        f'<div class="pf pf-raiz {clases}"{atributo}>{cabecera}{cuerpo}</div>'
        f"{_GUION}"
    )


def _a_html(contenido):
    """Convierte el contenido de un panel en HTML: otro componente, una
    tabla de pandas, una figura de matplotlib o texto (que se escapa)."""
    if isinstance(contenido, HTML):
        return contenido.data
    if hasattr(contenido, "columns") and hasattr(contenido, "iloc"):
        return tabla_html(contenido).data
    if hasattr(contenido, "savefig"):
        memoria = io.BytesIO()
        contenido.savefig(memoria, format="png", dpi=130, bbox_inches="tight", facecolor="white")
        datos = base64.b64encode(memoria.getvalue()).decode("ascii")
        return f'<img alt="Figura" src="data:image/png;base64,{datos}">'
    if isinstance(contenido, (list, tuple)):
        return "".join(_a_html(parte) for parte in contenido)
    return f'<div class="pf-parrafo">{_texto(contenido)}</div>'


def _cifra(valor, decimales, prefijo="", sufijo=""):
    """Cifra con su texto final ya escrito; el adorno la cuenta desde cero."""
    if not _es_numero(valor) or _falta(valor) or math.isinf(float(valor)):
        return f"{escape(prefijo)}{_texto(valor if not _falta(valor) else '—')}{escape(sufijo)}"
    es_entero = isinstance(valor, int) or (hasattr(valor, "dtype") and valor.dtype.kind in "iu")
    dec = 0 if es_entero else int(decimales)
    final = f"{prefijo}{formato_numero(valor, dec)}{sufijo}"
    return (f'<span data-pf-valor="{float(valor)!r}" data-pf-dec="{dec}" '
            f'data-pf-pre="{escape(prefijo, quote=True)}" data-pf-suf="{escape(sufijo, quote=True)}">'
            f"{escape(final)}</span>")


# ----------------------------------------------------------------------
# Componentes
# ----------------------------------------------------------------------
def encabezado_etapa(etapa, subtitulo=None):
    """Portada de una etapa: proyecto, título de la etapa y avance sobre
    `config.ETAPAS` (terminadas, actual, pendientes).
    Uso: `mostrar(encabezado_etapa("03_cribado_analisis", "Qué análisis merece profundizarse"))`."""
    if etapa not in ETAPAS:
        raise ValueError(f"Etapa desconocida: {etapa!r}")
    actual = ETAPAS.index(etapa)
    pasos_html = []
    for i, nombre in enumerate(ETAPAS):
        estado, descripcion = (("hecho", "terminada") if i < actual else
                               ("actual", "en curso") if i == actual else
                               ("pendiente", "pendiente"))
        rotulo = TITULOS_ETAPAS.get(nombre, nombre)
        pasos_html.append(
            f'<div class="pf-prog-paso pf-prog-{estado} pf-ap" style="--pf-i:{i + 2}" '
            f'title="Etapa {i + 1}: {escape(rotulo, quote=True)} ({descripcion})">'
            f'<span class="pf-prog-punto"></span>{i + 1}. {escape(rotulo)}</div>'
        )
    sub = f'<div class="pf-enc-sub pf-ap" style="--pf-i:1">{_texto(subtitulo)}</div>' if subtitulo else ""
    cuerpo = (
        f'<div class="pf-enc-rotulo pf-ap">{escape(ROTULO_PROYECTO)}</div>'
        f'<div class="pf-enc-proyecto pf-ap">{escape(TITULO_PROYECTO)}</div>'
        f'<div class="pf-enc-titulo pf-ap" style="--pf-i:1">'
        f"Etapa {actual + 1} de {len(ETAPAS)} · {escape(TITULOS_ETAPAS.get(etapa, etapa))}</div>"
        f'{sub}<div class="pf-progreso">{"".join(pasos_html)}</div>'
    )
    return _envolver(cuerpo, "encabezado", "pf-encabezado")


def tarjetas(lista, titulo=None, decimales=2):
    """Fila de tarjetas con cifras clave. Cada elemento es un diccionario
    con `titulo`, `valor` y, opcionalmente, `detalle`, `color`, `decimales`,
    `prefijo`, `sufijo`; o una tupla `(titulo, valor[, detalle])`.
    Uso: `mostrar(tarjetas([("Muestras", 40213), {"titulo": "No conformes", "valor": 6.4, "sufijo": " %", "color": "rojo"}]))`."""
    piezas = []
    for i, elemento in enumerate(lista):
        campo = _campos(elemento, ("titulo", "valor", "detalle"))
        cifra = _cifra(campo.get("valor"), campo.get("decimales", decimales),
                       str(campo.get("prefijo", "")), str(campo.get("sufijo", "")))
        detalle = (f'<div class="pf-tar-detalle">{_texto(campo["detalle"])}</div>'
                   if campo.get("detalle") not in (None, "") else "")
        piezas.append(
            f'<div class="pf-tarjeta pf-ap" style="--pf-i:{i};--pf-c:{_color(campo.get("color", "azul"))}">'
            f'<div class="pf-tar-titulo">{_texto(campo.get("titulo", ""))}</div>'
            f'<div class="pf-tar-valor">{cifra}</div>{detalle}</div>'
        )
    return _envolver(f'<div class="pf-tarjetas">{"".join(piezas)}</div>', "tarjetas", titulo=titulo)


def nota(texto, tipo="info", titulo=None, con_marcado=False):
    """Recuadro de texto destacado. `tipo`: `"info"`, `"hallazgo"`,
    `"advertencia"`, `"limite"` o `"simulacion"`. Con `con_marcado=True` el
    texto se inserta como HTML sin escapar (solo para marcado propio).
    Uso: `mostrar(nota(f"El modelo supera a la línea base en **{mejora:.1f} %**.", "hallazgo"))`."""
    if tipo not in TIPOS_NOTA:
        raise ValueError(f"Tipo de nota desconocido: {tipo!r}. Opciones: {sorted(TIPOS_NOTA)}")
    color, rotulo = TIPOS_NOTA[tipo]
    contenido = str(texto) if con_marcado else _texto(texto).replace("\n\n", "<br><br>")
    cuerpo = (f'<div class="pf-nota-titulo">{_texto(titulo or rotulo)}</div>'
              f'<div class="pf-nota-texto">{contenido}</div>')
    return _envolver(cuerpo, "nota", "pf-nota pf-ap", estilo=f"--pf-c:{COLORES[color]}")


def tabla_html(tabla, titulo=None, resaltar=None, decimales=4, columnas_resaltadas=None,
               barras=None, con_indice=False, maximo_filas=30, pie=None):
    """Tabla de pandas con formato uniforme (punto decimal, decimales fijos,
    enteros sin decimales).

    - `titulo`: leyenda; si no empieza por «Tabla» se le antepone.
    - `resaltar`: etiqueta(s) del índice de las filas a destacar, o una
      máscara booleana de la misma longitud que la tabla.
    - `decimales`: entero, o diccionario `{columna: decimales}`.
    - `columnas_resaltadas`: columna o lista de columnas con fondo tenue.
    - `barras`: columna(s) numérica(s) que llevan una barra dentro de la celda.
    - `maximo_filas`: filas mostradas como máximo (se avisa si se recorta).

    Uso: `mostrar(tabla_html(resultados, "Tabla 3. Validación cruzada", resaltar=resultados["RMSE"].idxmin(), barras="R2"))`."""
    total = len(tabla)
    mascara = _mascara_filas(tabla, resaltar)
    if maximo_filas is not None and total > maximo_filas:
        tabla, mascara = tabla.iloc[:maximo_filas], mascara[:maximo_filas]
    columnas = list(tabla.columns)
    destacadas = _lista(columnas_resaltadas)
    con_barra = [c for c in _lista(barras) if c in columnas]
    numericas = {c for c in columnas if getattr(tabla[c], "dtype", None) is not None
                 and tabla[c].dtype.kind in "iuf"}
    maximos = {}
    for c in con_barra:
        valores = [abs(float(v)) for v in tabla[c] if _es_numero(v) and not _falta(v)]
        maximos[c] = max(valores) if valores and max(valores) > 0 else 1.0

    def decimales_de(columna):
        if isinstance(decimales, dict):
            return decimales.get(columna, 4)
        return decimales

    def clases(columna):
        return ("pf-c" + (" pf-num" if columna in numericas else "")
                + (" pf-col-res" if columna in destacadas else ""))

    cabecera = "".join(f'<th class="{clases(c)}" scope="col">{escape(str(c))}</th>' for c in columnas)
    if con_indice:
        nombre = tabla.index.name if tabla.index.name is not None else ""
        cabecera = f'<th class="pf-c" scope="col">{escape(str(nombre))}</th>' + cabecera

    filas = []
    for posicion, (etiqueta, fila) in enumerate(zip(tabla.index, tabla.itertuples(index=False, name=None))):
        celdas = [f'<td class="pf-c">{escape(formato_numero(etiqueta, 0))}</td>'] if con_indice else []
        for columna, valor in zip(columnas, fila):
            texto = escape(formato_numero(valor, decimales_de(columna)))
            if columna in con_barra and _es_numero(valor) and not _falta(valor):
                ancho = 100 * abs(float(valor)) / maximos[columna]
                color = COLORES["rojo"] if float(valor) < 0 else COLORES["azul"]
                texto = (f'<span class="pf-celda-barra"><span class="pf-celda-pista">'
                         f'<span class="pf-celda-relleno" style="--pf-w:{ancho:.1f}%;--pf-c:{color};'
                         f'--pf-i:{min(posicion, 15)}"></span></span><span>{texto}</span></span>')
            celdas.append(f'<td class="{clases(columna)}">{texto}</td>')
        clase_fila = "pf-fila pf-ap" + (" pf-fila-res" if mascara[posicion] else "")
        filas.append(f'<tr class="{clase_fila}" style="--pf-i:{min(posicion, 15) * 0.5:g}">'
                     f'{"".join(celdas)}</tr>')

    leyenda = ""
    if titulo:
        titulo = str(titulo)
        if not titulo.lstrip().lower().startswith("tabla"):
            titulo = f"Tabla. {titulo}"
        leyenda = f'<caption class="pf-tabla-titulo">{_texto(titulo)}</caption>'
    pies = [pie] if pie else []
    if len(tabla) < total:
        pies.append(f"Se muestran las primeras {len(tabla)} de {total} filas.")
    pie_html = "".join(f'<div class="pf-tabla-pie">{_texto(p)}</div>' for p in pies)
    cuerpo = (f'<div class="pf-tabla-marco"><table class="pf-tabla">{leyenda}'
              f'<thead><tr class="pf-fila-cab">{cabecera}</tr></thead>'
              f'<tbody>{"".join(filas)}</tbody></table></div>{pie_html}')
    return _envolver(cuerpo, "tabla")


def _lista(valor):
    if valor is None:
        return []
    if isinstance(valor, (str, bytes)) or not hasattr(valor, "__iter__"):
        return [valor]
    return list(valor)


def _mascara_filas(tabla, resaltar):
    """Lista de booleanos, uno por fila, a partir de etiquetas o de una máscara."""
    total = len(tabla)
    if resaltar is None:
        return [False] * total
    valores = _lista(resaltar)
    es_mascara = (len(valores) == total and total > 0 and not isinstance(resaltar, (str, bytes))
                  and all(isinstance(v, bool) or (hasattr(v, "dtype") and v.dtype.kind == "b")
                          for v in valores))
    if es_mascara:
        return [bool(v) for v in valores]
    return [etiqueta in valores for etiqueta in tabla.index]


def barras_animadas(valores, titulo=None, formato=2, resaltar=None, color="azul", maximo=None):
    """Barras horizontales que crecen al aparecer. `valores` es un
    diccionario, una Serie o una lista de pares `(etiqueta, valor)`;
    `formato` son los decimales, una plantilla (`"{:.1f} %"`) o una función;
    `resaltar` es la etiqueta (o lista) que conserva el color pleno mientras
    las demás se atenúan; `maximo` fija el valor que ocupa todo el ancho.
    Uso: `mostrar(barras_animadas({"KNN": 0.71, "Árbol": 0.78}, "ROC-AUC en validación", formato=3, resaltar="Árbol"))`."""
    pares = _pares(valores)
    numeros = [abs(float(v)) for _, v in pares if _es_numero(v) and not _falta(v)]
    tope = float(maximo) if maximo else (max(numeros) if numeros and max(numeros) > 0 else 1.0)
    destacadas = _lista(resaltar)
    tono = _color(color)
    filas = []
    for i, (etiqueta, valor) in enumerate(pares):
        valido = _es_numero(valor) and not _falta(valor)
        ancho = min(100.0, 100 * abs(float(valor)) / tope) if valido else 0.0
        tinte = COLORES["rojo"] if valido and float(valor) < 0 else tono
        if destacadas and etiqueta not in destacadas:
            tinte = _mezclar(tinte, "#FFFFFF", 0.55)
        clase = " pf-bar-res" if etiqueta in destacadas else ""
        filas.append(
            f'<div class="pf-bar-etq{clase} pf-ap" style="--pf-i:{i}">{_texto(etiqueta)}</div>'
            f'<div class="pf-bar-pista"><div class="pf-bar-relleno" '
            f'style="--pf-w:{ancho:.1f}%;--pf-c:{tinte};--pf-i:{i}"></div></div>'
            f'<div class="pf-bar-valor pf-ap" style="--pf-i:{i}">'
            f"{escape(_aplicar_formato(valor, formato))}</div>"
        )
    return _envolver(f'<div class="pf-barras">{"".join(filas)}</div>', "barras", titulo=titulo)


def embudo(niveles, titulo=None, decimales=0):
    """Embudo de cribado: cada nivel es más estrecho que el anterior en
    proporción a su valor. Cada nivel es `(etiqueta, valor[, detalle])` o un
    diccionario con esas claves; `detalle` explica el paso al nivel siguiente.
    Uso: `mostrar(embudo([("Análisis candidatos", 16, "prueba rápida y uniforme"), ("Con ganancia sobre la línea base", 7), ("Elegidos", 3)]))`."""
    campos = [_campos(n, ("etiqueta", "valor", "detalle")) for n in niveles]
    numeros = [float(c["valor"]) for c in campos if _es_numero(c.get("valor")) and not _falta(c["valor"])]
    tope = max(numeros) if numeros and max(numeros) > 0 else 1.0
    piezas = []
    for i, campo in enumerate(campos):
        valor = campo.get("valor")
        valido = _es_numero(valor) and not _falta(valor)
        ancho = max(8.0, 100 * float(valor) / tope) if valido else 100.0
        tono = _mezclar(COLORES["azul"], COLORES["tinta"], i / max(len(campos) - 1, 1))
        piezas.append(
            f'<div class="pf-emb-nivel" style="--pf-w:{ancho:.1f}%;--pf-c:{tono};--pf-i:{i}">'
            f'<div class="pf-emb-valor">{_cifra(valor, decimales)}</div>'
            f'<div class="pf-emb-etq">{_texto(campo.get("etiqueta", ""))}</div></div>'
        )
        if i < len(campos) - 1:
            detalle = _texto(campo["detalle"]) if campo.get("detalle") else ""
            piezas.append(f'<div class="pf-emb-paso" style="--pf-i:{i}">{detalle}</div>')
    return _envolver(f'<div class="pf-embudo">{"".join(piezas)}</div>', "embudo", titulo=titulo)


def linea_tiempo(hitos, titulo=None):
    """Línea de tiempo horizontal. Cada hito es `(fecha, titulo[, detalle])`
    o un diccionario con esas claves y, opcionalmente, `color`.
    Uso: `mostrar(linea_tiempo([("2023-01", "Entrenamiento"), ("2025-07", "Validación"), ("2026-01", "Prueba", "se abre una sola vez")]))`."""
    piezas = []
    for i, hito in enumerate(hitos):
        campo = _campos(hito, ("fecha", "titulo", "detalle"))
        detalle = (f'<div class="pf-hito-det">{_texto(campo["detalle"])}</div>'
                   if campo.get("detalle") else "")
        piezas.append(
            f'<div class="pf-hito pf-ap" style="--pf-i:{i * 1.5:g};--pf-c:{_color(campo.get("color", "azul"))}">'
            f'<span class="pf-hito-punto"></span>'
            f'<div class="pf-hito-fecha">{_texto(formato_numero(campo.get("fecha", ""), 0))}</div>'
            f'<div class="pf-hito-titulo">{_texto(campo.get("titulo", ""))}</div>{detalle}</div>'
        )
    return _envolver(f'<div class="pf-linea-t">{"".join(piezas)}</div>', "linea", titulo=titulo)


def pestanas(contenidos, titulo=None):
    """Pestañas que funcionan sin JavaScript (botones de radio). `contenidos`
    es un diccionario `{rótulo: contenido}`; el contenido puede ser otro
    componente, una tabla de pandas, una figura de matplotlib, texto o una
    lista de ellos. Admite hasta 8 pestañas.
    Uso: `mostrar(pestanas({"Resumen": tarjetas(cifras), "Detalle": tabla_resultados, "Figura": fig}))`."""
    pares = _pares(contenidos)
    if not 1 <= len(pares) <= MAXIMO_PESTANAS:
        raise ValueError(f"Se admiten entre 1 y {MAXIMO_PESTANAS} pestañas (llegaron {len(pares)}).")
    paneles = [_a_html(contenido) for _, contenido in pares]
    _contador[0] += 1
    semilla = f"{_contador[0]}|" + "|".join(str(r) for r, _ in pares) + "|" + "".join(paneles)
    grupo = "pf-g" + hashlib.md5(semilla.encode("utf-8")).hexdigest()[:10]
    radios = "".join(
        f'<input class="pf-radio" type="radio" name="{grupo}" id="{grupo}-{i}"{" checked" if i == 0 else ""}>'
        for i in range(len(pares)))
    lenguetas = "".join(f'<label for="{grupo}-{i}">{_texto(rotulo)}</label>'
                        for i, (rotulo, _) in enumerate(pares))
    cuerpos = "".join(f'<div class="pf-panel">{panel}</div>' for panel in paneles)
    cuerpo = (f'<div class="pf-pestanas">{radios}<div class="pf-lenguetas">{lenguetas}</div>'
              f'<div class="pf-paneles">{cuerpos}</div></div>')
    return _envolver(cuerpo, "pestanas", titulo=titulo)


def pasos(lista, titulo=None, actual=None):
    """Lista numerada de pasos que aparecen en secuencia. Cada paso es un
    texto, una tupla `(titulo, detalle)` o un diccionario. Con `actual`
    (posición desde 1) los pasos posteriores se muestran como pendientes.
    Uso: `mostrar(pasos([("Ajustar", "solo con entrenamiento"), ("Decidir", "con validación"), ("Evaluar", "una vez, con prueba")]))`."""
    piezas = []
    for i, paso in enumerate(lista, start=1):
        campo = _campos(paso, ("titulo", "detalle"))
        estado = ""
        if actual is not None:
            estado = " pf-paso-act" if i == actual else " pf-paso-pend" if i > actual else ""
        detalle = (f'<div class="pf-paso-det">{_texto(campo["detalle"])}</div>'
                   if campo.get("detalle") else "")
        piezas.append(
            f'<div class="pf-paso{estado} pf-ap" style="--pf-i:{(i - 1) * 2}">'
            f'<div class="pf-paso-num">{i}</div>'
            f'<div><div class="pf-paso-titulo">{_texto(campo.get("titulo", ""))}</div>{detalle}</div></div>'
        )
    return _envolver(f'<div class="pf-pasos">{"".join(piezas)}</div>', "pasos", titulo=titulo)


def comparacion(opciones, titulo=None, elegida=None, sello="Elegida", decimales=2):
    """Opciones lado a lado (línea base contra modelo, antes contra después).
    Cada opción es un diccionario con `titulo`, `valor`, y opcionalmente
    `etiqueta` (qué mide el valor), `puntos` (lista de frases), `decimales`,
    `prefijo`, `sufijo`. `elegida` es el título o la posición (desde 0) de la
    opción que se destaca con el sello.
    Uso: `mostrar(comparacion([{"titulo": "Línea base", "valor": 12.4, "etiqueta": "RMSE"}, {"titulo": "Árbol podado", "valor": 9.1, "etiqueta": "RMSE"}], elegida="Árbol podado"))`."""
    piezas = []
    for i, opcion in enumerate(opciones):
        campo = _campos(opcion, ("titulo", "valor", "etiqueta"))
        es_elegida = elegida is not None and elegida in (i, campo.get("titulo")) \
            and not (isinstance(elegida, bool))
        marca = f'<div class="pf-op-sello">{_texto(sello)}</div>' if es_elegida else ""
        cifra = _cifra(campo.get("valor"), campo.get("decimales", decimales),
                       str(campo.get("prefijo", "")), str(campo.get("sufijo", "")))
        etiqueta = (f'<div class="pf-op-etq">{_texto(campo["etiqueta"])}</div>'
                    if campo.get("etiqueta") else "")
        puntos = "".join(f'<div class="pf-op-punto">{_texto(p)}</div>'
                         for p in _lista(campo.get("puntos")))
        puntos = f'<div class="pf-op-puntos">{puntos}</div>' if puntos else ""
        piezas.append(
            f'<div class="pf-opcion{" pf-op-elegida" if es_elegida else ""} pf-ap" style="--pf-i:{i * 2}">'
            f'{marca}<div class="pf-op-titulo">{_texto(campo.get("titulo", ""))}</div>'
            f'<div class="pf-op-valor">{cifra}</div>{etiqueta}{puntos}</div>'
        )
    return _envolver(f'<div class="pf-comparacion">{"".join(piezas)}</div>', "comparacion", titulo=titulo)
