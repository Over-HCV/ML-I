"""Capa 7 · Componentes HTML para presentar resultados.

Responsabilidad única: convertir resultados ya calculados en piezas visuales
con aspecto de diapositiva (encabezados, tarjetas, tablas, barras, embudo,
línea de tiempo, pestañas) y en las piezas de la presentación (diapositiva,
cifras grandes, flujo con flechas, matriz de confusión, línea en SVG,
columnas, fichas, fórmula, recuadros). Cada función devuelve un objeto
`IPython.display.HTML`; quien llama lo pasa a `salida.mostrar`.

Criterios de diseño (sistema editorial, el mismo de las figuras):

- Página de papel claro, sin cajas, sin bordes de color a un lado, sin
  sombras ni píldoras: la jerarquía sale de la tipografía (titulares y
  cifras grandes en Fraunces, texto y rótulos en IBM Plex Sans con
  versalitas, cifras tabulares y código en JetBrains Mono), de filetes finos
  y del espacio en blanco; las notas son anotaciones al margen. El color se
  reserva para los datos.
- Cada pieza lleva su propia hoja de estilos, porque Colab dibuja cada salida
  en un marco aislado; cada hoja importa las fuentes de Google Fonts y, sin
  conexión, cae en Georgia, la sans del sistema y `ui-monospace`. Las clases
  usan el prefijo `pf-`.
- Movimiento sobrio y solo con CSS (aparecer, filetes que se trazan, barras
  que crecen); el guion final es un adorno (conteo de cifras y espera a que
  la pieza entre en pantalla): si el visor no lo ejecuta, el resultado es
  igualmente correcto. Se respeta `prefers-reduced-motion`.
- Fondo y colores de texto propios, legibles en temas claros y oscuros.
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
                          "Aprendizaje Automático de Máquina I, proyecto final")
TITULO_PROYECTO = getattr(config, "TITULO_PROYECTO",
                          "Aprendizaje automático sobre la base de muestras de un "
                          "laboratorio de microbiología")

TENUE = "#5B6470"      # texto secundario (contraste suficiente sobre el papel)
LINEA = "#D8D4CA"      # filetes finos
PAPEL = "#FCFBF8"      # fondo de papel
TINTA = "#1B1F24"
# Colores de la matriz de confusión (VN, FP, FN, VP): FP y FN son los de la ecuación del costo y VP el de los
# verdaderos positivos de Bayes (`graficos_presentacion.TINTAS`); VN, un neutro cálido.
COLORES_CONFUSION = ("#A39C8C", "#6B6B6B", "#B8423F", "#2F8A4E")
TIPOS_NOTA = {
    "info": ("azul", "Nota"),
    "hallazgo": ("verde", "Hallazgo"),
    "advertencia": ("naranja", "Advertencia"),
    "limite": ("rojo", "Límite"),
    "simulacion": ("morado", "Datos simulados"),
}
MAXIMO_PESTANAS = 8
_contador = [0]

# Tipografía del sistema editorial: Fraunces (titulares y cifras grandes), IBM Plex Sans
# (texto y rótulos) y JetBrains Mono (cifras tabulares y código). Se cargan de Google Fonts
# dentro de cada hoja (Colab aísla cada salida en su propio marco) y, sin conexión, caen en
# Georgia, la sans del sistema y `ui-monospace`.
FUENTES_CSS = ("@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght,SOFT,WONK@9..144,"
               "400..700,0..100,0..1&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400"
               "&family=JetBrains+Mono:wght@400;500;600&display=swap');")
SERIF = '"Fraunces","Iowan Old Style","Palatino Linotype",Georgia,"Times New Roman",serif'
SANS = '"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif'
MONO = '"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace'


# ----------------------------------------------------------------------
# Hojas de estilo (una base común y una por componente)
# ----------------------------------------------------------------------
_CURVA = "cubic-bezier(.2,.7,.2,1)"
_CSS = {}
_CSS["base"] = f"""{FUENTES_CSS}
.pf{{--pf-azul:{COLORES['azul']};--pf-rojo:{COLORES['rojo']};--pf-verde:{COLORES['verde']};
--pf-naranja:{COLORES['naranja']};--pf-morado:{COLORES['morado']};--pf-gris:{COLORES['gris']};
--pf-tinta:{TINTA};--pf-fondo:#F3F0E8;--pf-papel:{PAPEL};--pf-tenue:{TENUE};--pf-linea:{LINEA};
--pf-serif:{SERIF};--pf-sans:{SANS};--pf-mono:{MONO};
box-sizing:border-box;position:relative;max-width:980px;margin:10px 0;padding:18px 22px;
background:var(--pf-papel);color:var(--pf-tinta);border:0;border-radius:0;
font-family:var(--pf-sans);font-size:15px;line-height:1.5;text-align:left;
-webkit-font-smoothing:antialiased;font-feature-settings:"kern" 1}}
.pf *{{box-sizing:border-box}}
.pf .pf:not(.pf-nota):not(.pf-encabezado){{padding:0;margin:0;background:transparent;max-width:none}}
.pf .pf-titulo{{font-family:var(--pf-sans);font-size:11.5px;font-weight:600;letter-spacing:.09em;
text-transform:uppercase;margin:0 0 12px;color:var(--pf-tenue)}}
.pf .pf-titulo::after{{content:"";display:block;height:1px;background:var(--pf-linea);margin-top:6px;
transform-origin:left;animation:pf-crecer .9s {_CURVA} both;animation-delay:var(--pf-r,0ms)}}
.pf code{{font-family:var(--pf-mono);font-size:.86em;background:transparent;color:var(--pf-tinta);padding:0}}
.pf strong{{font-weight:600;color:inherit}}
.pf .pf-num,.pf [data-pf-valor]{{font-variant-numeric:tabular-nums lining-nums}}
@keyframes pf-aparecer{{from{{opacity:0;transform:translateY(6px)}}}}
@keyframes pf-crecer{{from{{transform:scaleX(0)}}}}
@keyframes pf-abrir{{from{{transform:scaleX(0)}}}}
.pf .pf-ap{{animation:pf-aparecer .6s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*90ms + var(--pf-r,0ms))}}
.pf.pf-pausa,.pf.pf-pausa *{{animation-play-state:paused!important}}
@media (prefers-reduced-motion:reduce){{.pf,.pf *{{animation:none!important;transition:none!important}}}}
"""
_CSS["encabezado"] = f"""
.pf.pf-encabezado{{padding:22px 24px 18px}}
.pf-enc-rotulo{{font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--pf-tenue);font-weight:600}}
.pf-enc-proyecto{{font-size:14px;color:var(--pf-tenue);margin-top:2px;font-style:italic}}
.pf-enc-titulo{{font-family:var(--pf-serif);font-size:34px;font-weight:560;line-height:1.1;margin:14px 0 6px;
letter-spacing:-.01em;font-variation-settings:"SOFT" 0,"WONK" 0;font-optical-sizing:auto}}
.pf-enc-sub{{font-size:16px;color:var(--pf-tenue);max-width:72ch}}
.pf-progreso{{display:flex;margin-top:20px;border-top:1px solid var(--pf-tinta);padding-top:8px;
transform-origin:left;animation:pf-crecer 1s {_CURVA} both}}
.pf-prog-paso{{flex:1 1 0;min-width:0;padding:0 8px 0 0;font-size:11.5px;line-height:1.3;color:#9A958A}}
.pf-prog-punto{{display:block;font-family:var(--pf-mono);font-size:11px;margin-bottom:2px}}
.pf-prog-hecho{{color:var(--pf-tenue)}}
.pf-prog-actual{{color:var(--pf-tinta);font-weight:600}}
.pf-prog-actual .pf-prog-punto{{color:var(--pf-azul)}}
"""
_CSS["tarjetas"] = f"""
.pf-tarjetas{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:18px 26px}}
.pf-tarjeta{{border-top:1px solid var(--pf-tinta);padding-top:9px}}
.pf-tar-titulo{{font-size:11.5px;color:var(--pf-tenue);text-transform:uppercase;letter-spacing:.08em;font-weight:600}}
.pf-tar-titulo::before{{content:"";display:inline-block;width:7px;height:7px;background:var(--pf-c,var(--pf-azul));
margin-right:7px;vertical-align:1px}}
.pf-tar-valor{{font-family:var(--pf-serif);font-size:34px;font-weight:560;line-height:1.1;margin:6px 0 2px;
color:var(--pf-tinta);font-variant-numeric:tabular-nums lining-nums;letter-spacing:-.01em}}
.pf-tar-detalle{{font-size:13px;color:var(--pf-tenue)}}
"""
_CSS["nota"] = """
.pf.pf-nota{display:grid;grid-template-columns:118px minmax(0,1fr);column-gap:20px;padding:12px 22px 12px}
.pf.pf-nota::before{content:"";position:absolute;left:22px;right:22px;top:0;height:1px;background:var(--pf-linea)}
.pf-nota-titulo{font-size:11.5px;font-weight:600;text-transform:uppercase;letter-spacing:.09em;line-height:1.35;
color:var(--pf-o,var(--pf-tenue));padding-top:3px}
.pf-nota-texto{color:var(--pf-tinta);max-width:78ch}
.pf-diapo .pf.pf-nota{padding:10px 0 0}
.pf-diapo .pf.pf-nota::before{left:0;right:0}
@media (max-width:640px){.pf.pf-nota{grid-template-columns:1fr}}
"""
_CSS["tabla"] = f"""
.pf-tabla-marco{{overflow-x:auto}}
.pf .pf-tabla{{border-collapse:collapse;width:auto;min-width:min(100%,460px);margin:0;border:0;font-size:14px;
color:var(--pf-tinta);background:transparent;table-layout:auto;border-top:1.5px solid var(--pf-tinta);
border-bottom:1.5px solid var(--pf-tinta)}}
.pf .pf-tabla caption.pf-tabla-titulo{{caption-side:top;text-align:left;font-family:var(--pf-serif);font-size:17px;
font-weight:560;color:var(--pf-tinta);padding:0 0 10px}}
.pf .pf-tabla thead tr.pf-fila-cab{{background:transparent}}
.pf .pf-tabla thead th.pf-c{{padding:7px 12px;text-align:left;font-weight:600;font-size:12px;letter-spacing:.04em;
color:var(--pf-tenue);background:transparent;border:0;border-bottom:1px solid var(--pf-tinta);white-space:nowrap}}
.pf .pf-tabla tbody tr.pf-fila{{background:transparent}}
.pf .pf-tabla tbody tr.pf-fila:hover{{background:var(--pf-fondo)}}
.pf .pf-tabla tbody td.pf-c{{padding:6px 12px;text-align:left;vertical-align:middle;color:var(--pf-tinta);
border:0;border-bottom:1px solid #ECE8DF}}
.pf .pf-tabla tbody tr:last-child td.pf-c{{border-bottom:0}}
.pf .pf-tabla thead th.pf-num,.pf .pf-tabla tbody td.pf-num{{text-align:right;font-family:var(--pf-mono);
font-size:13px;font-variant-numeric:tabular-nums}}
.pf .pf-tabla thead th.pf-num{{font-family:var(--pf-sans);font-size:12px}}
.pf .pf-tabla thead th.pf-col-res,.pf .pf-tabla tbody td.pf-col-res{{background:#F4F1EA}}
.pf .pf-tabla tbody tr.pf-fila-res,.pf .pf-tabla tbody tr.pf-fila-res:hover{{background:#EFE9DA}}
.pf .pf-tabla tbody tr.pf-fila-res td.pf-c{{font-weight:600}}
.pf-celda-barra{{display:flex;align-items:center;justify-content:flex-end;gap:10px}}
.pf-celda-pista{{flex:1 1 70px;min-width:70px;max-width:180px;height:6px;background:transparent;overflow:hidden}}
.pf-celda-relleno{{display:block;height:100%;width:var(--pf-w,0%);background:var(--pf-c,var(--pf-azul));
transform-origin:left;animation:pf-crecer .8s {_CURVA} both;
animation-delay:calc(var(--pf-i,0)*60ms + var(--pf-r,0ms))}}
.pf-tabla-pie{{font-size:12.5px;color:var(--pf-tenue);margin-top:8px;font-style:italic}}
"""
_CSS["barras"] = f"""
.pf-barras{{display:grid;grid-template-columns:fit-content(40%) 1fr max-content;gap:7px 12px;align-items:center}}
.pf-bar-etq{{font-size:14px;text-align:right;color:var(--pf-tenue)}}
.pf-bar-pista{{height:12px;border-left:1px solid var(--pf-tinta)}}
.pf-bar-relleno{{height:100%;width:var(--pf-w,0%);background:var(--pf-c,var(--pf-azul));
transform-origin:left;animation:pf-crecer .9s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*80ms + var(--pf-r,0ms))}}
.pf-bar-valor{{font-family:var(--pf-mono);font-size:13px;font-variant-numeric:tabular-nums;color:var(--pf-tinta)}}
.pf-bar-res{{font-weight:600;color:var(--pf-tinta)}}
"""
_CSS["embudo"] = f"""
.pf-embudo{{display:flex;flex-direction:column;align-items:flex-start}}
.pf-emb-nivel{{width:var(--pf-w,100%);min-width:200px;max-width:100%;border-top:3px solid var(--pf-c,var(--pf-tinta));
padding:6px 0 2px;transform-origin:left;animation:pf-abrir .8s {_CURVA} both;
animation-delay:calc(var(--pf-i,0)*240ms + var(--pf-r,0ms))}}
.pf-emb-valor{{font-family:var(--pf-serif);font-size:30px;font-weight:560;line-height:1.05;color:var(--pf-tinta);
font-variant-numeric:tabular-nums lining-nums;display:inline-block;margin-right:10px}}
.pf-emb-etq{{font-size:14px;color:var(--pf-tinta);display:inline}}
.pf-emb-paso{{font-size:13px;color:var(--pf-tenue);padding:4px 0 8px 2px;max-width:60ch;font-style:italic;
animation:pf-aparecer .5s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*240ms + 130ms + var(--pf-r,0ms))}}
.pf-emb-paso::before{{content:"\\2193\\00a0";font-style:normal;color:var(--pf-gris)}}
"""
_CSS["linea"] = """
.pf-linea-t{display:flex}
.pf-hito{flex:1 1 0;min-width:0;position:relative;padding:20px 8px 0 0;text-align:left}
.pf-hito::before{content:"";position:absolute;top:4px;left:0;right:0;height:1px;background:var(--pf-tinta)}
.pf-hito-punto{position:absolute;top:0;left:0;width:9px;height:9px;border-radius:50%;
background:var(--pf-c,var(--pf-tinta))}
.pf-hito-fecha{font-family:var(--pf-mono);font-size:11px;color:var(--pf-tenue)}
.pf-hito-titulo{font-size:14px;font-weight:600;color:var(--pf-tinta);line-height:1.25}
.pf-hito-det{font-size:12.5px;color:var(--pf-tenue);line-height:1.35;margin-top:2px}
"""
_CSS["pasos"] = """
.pf-paso{display:grid;grid-template-columns:38px 1fr;column-gap:12px;padding:8px 0 10px;border-top:1px solid var(--pf-linea)}
.pf-paso-num{font-family:var(--pf-serif);font-size:24px;font-weight:500;line-height:1;color:var(--pf-tenue);
font-variant-numeric:oldstyle-nums}
.pf-paso-titulo{font-weight:600;color:var(--pf-tinta)}
.pf-paso-det{font-size:14px;color:var(--pf-tenue);max-width:78ch}
.pf-paso-pend .pf-paso-titulo,.pf-paso-pend .pf-paso-num{color:#A8A398}
.pf-paso-act .pf-paso-num{color:var(--pf-azul)}
"""
_CSS["pestanas"] = """
.pf-pestanas>input.pf-radio{position:absolute;opacity:0;pointer-events:none}
.pf-lenguetas{display:flex;flex-wrap:wrap;gap:2px 18px;border-bottom:1px solid var(--pf-tinta);margin-bottom:16px}
.pf-lenguetas label{padding:6px 0;margin-bottom:-1px;font-size:12px;font-weight:600;letter-spacing:.08em;
text-transform:uppercase;color:var(--pf-tenue);cursor:pointer;border-bottom:2px solid transparent;transition:color .2s,border-color .2s}
.pf-lenguetas label:hover{color:var(--pf-tinta)}
.pf-paneles>.pf-panel{display:none}
.pf-panel img{max-width:100%;height:auto;display:block}
""" + "".join(
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):checked~.pf-lenguetas label:nth-of-type({k})"
    "{color:var(--pf-tinta);border-bottom-color:var(--pf-tinta)}\n"
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):focus-visible~.pf-lenguetas label:nth-of-type({k})"
    "{outline:2px solid var(--pf-azul);outline-offset:2px}\n"
    f".pf-pestanas>input.pf-radio:nth-of-type({k}):checked~.pf-paneles>.pf-panel:nth-of-type({k})"
    "{display:block;animation:pf-aparecer .4s ease-out both}\n"
    for k in range(1, MAXIMO_PESTANAS + 1)
)
_CSS["comparacion"] = """
.pf-comparacion{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px 28px}
.pf-opcion{position:relative;border-top:1px solid var(--pf-linea);padding-top:9px}
.pf-opcion.pf-op-elegida{border-top:2px solid var(--pf-tinta);padding-top:8px}
.pf-op-sello{font-size:11px;font-weight:600;letter-spacing:.1em;text-transform:uppercase;color:#2F6A3E;float:right}
.pf-op-titulo{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--pf-tenue);font-weight:600}
.pf-op-valor{font-family:var(--pf-serif);font-size:38px;font-weight:560;line-height:1.1;margin-top:4px;color:var(--pf-tinta);
font-variant-numeric:tabular-nums lining-nums;letter-spacing:-.01em}
.pf-opcion:not(.pf-op-elegida) .pf-op-valor{color:#6E7781}
.pf-op-etq{font-size:13px;color:var(--pf-tenue)}
.pf-op-puntos{margin-top:8px;font-size:14px;color:var(--pf-tinta)}
.pf-op-punto{margin:3px 0}
.pf-op-punto::before{content:"\\203A\\00a0";color:var(--pf-gris)}
"""
# --- Piezas de la presentación (diapositivas) ---
_CSS["diapositiva"] = f"""
.pf.pf-diapo{{max-width:1100px;min-height:400px;padding:22px 34px 16px;font-size:16px;overflow:hidden;
animation:pf-aparecer .6s {_CURVA} both}}
.pf-dia-barra{{position:absolute;top:16px;left:34px;right:34px;height:1px;background:var(--pf-linea)}}
.pf-dia-barra::after{{content:"";position:absolute;left:0;top:-1px;height:3px;width:var(--pf-w,0%);
background:var(--pf-tinta);transform-origin:left;animation:pf-crecer 1.2s {_CURVA} both}}
.pf-dia-cab{{display:flex;align-items:baseline;justify-content:space-between;gap:18px;margin:14px 0 18px}}
.pf-dia-rotulo{{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--pf-tenue);font-weight:600}}
.pf-dia-titulo{{font-family:var(--pf-serif);font-size:38px;font-weight:560;line-height:1.08;margin-top:8px;
color:var(--pf-tinta);letter-spacing:-.015em;font-variation-settings:"SOFT" 0,"WONK" 0;font-optical-sizing:auto;
max-width:30ch}}
.pf-dia-sub{{font-size:17px;color:var(--pf-tenue);margin-top:8px;max-width:72ch;font-style:italic}}
.pf-dia-num{{font-family:var(--pf-mono);font-size:12px;color:var(--pf-tenue);white-space:nowrap}}
.pf-dia-cuerpo{{display:flex;flex-direction:column;gap:22px}}
.pf-dia-bloque{{animation:pf-aparecer .6s {_CURVA} both;animation-delay:var(--pf-r,0ms)}}
.pf-dia-pie{{margin-top:22px;padding-top:8px;border-top:1px solid var(--pf-linea);font-size:12px;
color:var(--pf-tenue);display:flex;justify-content:space-between;gap:14px;flex-wrap:wrap;letter-spacing:.02em}}
.pf.pf-diapo.pf-dia-portada{{display:flex;flex-direction:column;justify-content:center;min-height:500px;
padding:44px 52px 22px}}
.pf-dia-portada .pf-dia-titulo{{font-size:54px;max-width:20ch;line-height:1.03;font-weight:540}}
.pf-dia-portada .pf-dia-sub{{font-size:21px;margin-top:16px}}
.pf-dia-portada .pf-dia-cuerpo{{margin-top:22px}}
.pf-dia-portada .pf-dia-barra{{left:52px;right:52px;top:22px}}
.pf-diapo .pf-nota-texto{{max-width:none}}
.pf-diapo .pf-col>.pf-raiz~.pf-raiz{{margin-top:20px!important}}
.pf-diapo .pf-col .pf-tarjetas{{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}}
.pf-diapo .pf-parrafo{{max-width:70ch;font-size:17px;line-height:1.55}}
"""
_CSS["heroe"] = """
.pf-heroes{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:18px 30px}
.pf-heroe{border-top:1px solid var(--pf-tinta);padding-top:6px}
.pf-heroe-valor{font-family:var(--pf-serif);font-size:64px;font-weight:540;line-height:1;color:var(--pf-o);
font-variant-numeric:tabular-nums lining-nums;letter-spacing:-.025em;font-variation-settings:"SOFT" 0,"WONK" 0}
.pf-heroe-etq{font-size:15px;font-weight:600;margin-top:6px;color:var(--pf-tinta)}
.pf-heroe-det{font-size:13.5px;color:var(--pf-tenue);margin-top:3px;max-width:46ch}
"""
_CSS["flujo"] = f"""
.pf-flujo{{display:flex;align-items:stretch}}
.pf-flujo-paso{{flex:1 1 0;min-width:0;border-top:1px solid var(--pf-tinta);padding:8px 2px 0;
animation:pf-aparecer .5s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*240ms + var(--pf-r,0ms))}}
.pf-flujo-marca{{font-family:var(--pf-mono);font-size:11px;color:var(--pf-o)}}
.pf-flujo-titulo{{font-family:var(--pf-serif);font-weight:560;font-size:19px;line-height:1.2;margin-top:2px;color:var(--pf-tinta)}}
.pf-flujo-det{{font-size:13.5px;color:var(--pf-tenue);margin-top:3px}}
.pf-flujo-lleno .pf-flujo-titulo{{color:var(--pf-o)}}
.pf-flujo-lleno{{border-top:3px solid var(--pf-o);padding-top:6px}}
.pf-flujo-flecha{{flex:0 0 34px;display:flex;align-items:flex-start;justify-content:center;padding-top:16px}}
.pf-flujo-flecha svg{{width:24px;height:14px;overflow:visible}}
.pf-trazo{{stroke-dasharray:1;stroke-dashoffset:0;animation:pf-trazar .5s ease-out both;
animation-delay:calc(var(--pf-i,0)*240ms + 160ms + var(--pf-r,0ms))}}
@keyframes pf-trazar{{from{{stroke-dashoffset:1}}}}
@media (max-width:760px){{.pf-flujo{{flex-direction:column}}.pf-flujo-flecha{{flex-basis:22px;transform:rotate(90deg)}}}}
"""
_CSS["matriz"] = f"""
.pf-mc{{display:grid;grid-template-columns:max-content 1fr 1fr;gap:0;align-items:stretch;max-width:520px;
border-top:1.5px solid var(--pf-tinta);border-bottom:1.5px solid var(--pf-tinta)}}
.pf-mc-eje{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--pf-tenue);font-weight:600;
text-align:left;padding:6px 12px 5px;border-bottom:1px solid var(--pf-tinta)}}
.pf-mc>div:first-child{{border-bottom:1px solid var(--pf-tinta)}}
.pf-mc-fila{{font-size:13px;font-weight:600;color:var(--pf-tinta);align-self:center;text-align:right;padding:0 14px 0 0}}
.pf-mc-celda{{padding:10px 12px;border-left:1px solid var(--pf-linea);
animation:pf-aparecer .5s {_CURVA} both;animation-delay:calc(var(--pf-i,0)*160ms + var(--pf-r,0ms))}}
.pf-mc-celda:nth-child(n+7){{border-top:1px solid var(--pf-linea)}}
.pf-mc-valor{{font-family:var(--pf-serif);font-size:32px;font-weight:560;line-height:1.05;color:var(--pf-c);
font-variant-numeric:tabular-nums lining-nums}}
.pf-mc-etq{{font-size:12.5px;color:var(--pf-tenue);margin-top:2px}}
.pf-mc-pie{{font-size:12px;color:var(--pf-tenue);margin-top:8px;font-style:italic}}
"""
_CSS["linea_svg"] = """
.pf-lsvg svg{display:block;width:100%;height:auto;overflow:visible}
.pf-lsvg text{font-family:var(--pf-mono);fill:#5B6470}
.pf-lsvg text.pf-lsvg-etq{font-family:var(--pf-sans)}
.pf-lsvg .pf-lsvg-serie{fill:none;stroke-width:2.2;stroke-linejoin:round;stroke-linecap:round}
.pf-lsvg .pf-trazo{stroke-dasharray:1;animation:pf-trazar 1.4s ease-out both;
animation-delay:calc(var(--pf-i,0)*90ms + var(--pf-r,0ms))}
@keyframes pf-trazar{from{stroke-dashoffset:1}}
.pf-lsvg-leyenda{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--pf-tinta);margin-top:6px}
.pf-lsvg-muestra{display:inline-block;width:16px;height:2px;margin-right:6px;vertical-align:middle;background:var(--pf-c)}
"""
_CSS["columnas"] = """
.pf-columnas{display:grid;gap:28px;align-items:start}
.pf-columnas>.pf-col{min-width:0}
@media (max-width:760px){.pf-columnas{grid-template-columns:1fr!important}}
"""
_CSS["fichas"] = """
.pf-fichas{display:flex;flex-wrap:wrap;row-gap:4px;font-size:14.5px;color:var(--pf-tinta)}
.pf-ficha{display:inline-block}
.pf-ficha:not(:last-child)::after{content:",";color:var(--pf-gris);margin-right:10px}
"""
_CSS["formula"] = """
.pf-formula{display:flex;align-items:center;flex-wrap:wrap;gap:12px;font-family:var(--pf-serif);font-size:22px;
color:var(--pf-tinta);padding:6px 0;font-style:italic}
.pf-f-izq{font-weight:600;font-style:normal}
.pf-f-frac{display:inline-flex;flex-direction:column;align-items:center;text-align:center}
.pf-f-num{padding:0 8px 4px;border-bottom:1px solid var(--pf-tinta)}
.pf-f-den{padding:4px 8px 0}
.pf-f-det{font-family:var(--pf-sans);font-style:normal;font-size:14px;color:var(--pf-tenue);flex-basis:100%}
.pf-f-a{color:#2F6A3E;font-weight:600}.pf-f-b{color:#A63D40;font-weight:600}
"""
_CSS["recuadros"] = """
.pf-recuadros{display:grid;gap:18px 30px}
.pf-recuadro{display:grid;grid-template-columns:44px 1fr;column-gap:12px;align-items:start;
border-top:1px solid var(--pf-tinta);padding-top:10px}
.pf-rec-marca{font-family:var(--pf-serif);font-size:40px;font-weight:500;line-height:.95;color:var(--pf-c);
font-variant-numeric:lining-nums}
.pf-rec-titulo{font-family:var(--pf-serif);font-weight:560;font-size:20px;line-height:1.2;color:var(--pf-tinta)}
.pf-rec-texto{font-size:14.5px;color:#3B4148;margin-top:5px}
@media (max-width:760px){.pf-recuadros{grid-template-columns:1fr!important}}
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
    enteros sin decimales y celda en blanco para los faltantes.
    Uso: `formato_numero(0.81234, 3)` → `'0.812'`."""
    if _falta(valor):
        return ""
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
        return "" if _falta(valor) else formato.format(valor)
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


def _envolver(cuerpo, componente, clases="", estilo="", titulo=None, hoja_extra=""):
    """Raíz de un componente: estilos propios + contenido + adorno opcional."""
    hoja = _CSS["base"] + _CSS.get(componente, "") + hoja_extra
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
        return f"{escape(prefijo)}{_texto(valor if not _falta(valor) else '')}{escape(sufijo)}"
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
    Uso: `mostrar(encabezado_etapa("03_cribado_regresion", "Qué análisis merece profundizarse"))`."""
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
            f'<span class="pf-prog-punto">{i + 1:02d}</span>{escape(rotulo)}</div>'
        )
    sub = f'<div class="pf-enc-sub pf-ap" style="--pf-i:1">{_texto(subtitulo)}</div>' if subtitulo else ""
    cuerpo = (
        f'<div class="pf-enc-rotulo pf-ap">{escape(ROTULO_PROYECTO)}</div>'
        f'<div class="pf-enc-proyecto pf-ap">{escape(TITULO_PROYECTO)}</div>'
        f'<div class="pf-enc-titulo pf-ap" style="--pf-i:1">'
        f"Etapa {actual + 1} de {len(ETAPAS)}: {escape(TITULOS_ETAPAS.get(etapa, etapa))}</div>"
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
    return _envolver(cuerpo, "nota", "pf-nota pf-ap", estilo=f"--pf-c:{COLORES[color]};--pf-o:{_oscuro(color)}")


def tabla_html(tabla, titulo=None, resaltar=None, decimales=4, columnas_resaltadas=None,
               barras=None, con_indice=False, maximo_filas=30, pie=None):
    """Tabla de pandas con formato uniforme (punto decimal, decimales fijos,
    enteros sin decimales).

    - `titulo`: leyenda; si no empieza por "Tabla" se le antepone.
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


# ----------------------------------------------------------------------
# Piezas de la presentación
# ----------------------------------------------------------------------
def _oscuro(color):
    """Variante oscura de un color, para texto con contraste suficiente sobre fondo claro."""
    return _mezclar(_color(color), COLORES["tinta"], 0.35)


def _tenue(color):
    """Variante muy clara de un color, para fondos."""
    return _mezclar(_color(color), "#FFFFFF", 0.86)


def diapositiva(titulo, cuerpo, numero=None, total=None, pie=None, rotulo=None, subtitulo=None,
                portada=False, pausa=350):
    """Diapositiva de ancho completo: rótulo de sección, título, contador
    `numero / total` con barra de avance, cuerpo y pie. `cuerpo` es una lista
    de piezas (otros componentes, tablas, texto); cada pieza entra `pausa` ms
    después de la anterior, y sus propias animaciones esperan ese turno.
    Con `portada=True` usa fondo oscuro y título grande.
    Uso: `mostrar(diapositiva("Regresión", [tarjetas(cifras), nota(texto)], numero=6, total=10))`."""
    bloques = "".join(
        f'<div class="pf-dia-bloque" style="--pf-r:{(i + 1) * int(pausa)}ms">{_a_html(pieza)}</div>'
        for i, pieza in enumerate(_lista(cuerpo)))
    # Una sola hoja de estilos y un solo adorno para toda la diapositiva (las piezas anidadas traen los suyos).
    hojas = []
    for hoja in re.findall(r"<style>(.*?)</style>", bloques, flags=re.S):
        hoja = hoja.replace(_CSS["base"], "")
        if hoja not in hojas:
            hojas.append(hoja)
    bloques = re.sub(r"<style>.*?</style>", "", bloques, flags=re.S).replace(_GUION, "")
    contador, barra = "", ""
    if numero is not None:
        contador = f'<div class="pf-dia-num">{int(numero):02d}' + (f" / {int(total):02d}" if total else "") + "</div>"
        if total:
            barra = f'<div class="pf-dia-barra" style="--pf-w:{100 * int(numero) / int(total):.1f}%"></div>'
    rot = f'<div class="pf-dia-rotulo">{_texto(rotulo)}</div>' if rotulo else ""
    sub = f'<div class="pf-dia-sub">{_texto(subtitulo)}</div>' if subtitulo else ""
    partes = [_texto(p) for p in _lista(pie)]
    pie_html = (f'<div class="pf-dia-pie">{"".join(f"<span>{p}</span>" for p in partes)}</div>'
                if partes else "")
    cuerpo_html = (f'{barra}<div class="pf-dia-cab"><div>{rot}<div class="pf-dia-titulo">{_texto(titulo)}</div>'
                   f'{sub}</div>{contador}</div><div class="pf-dia-cuerpo">{bloques}</div>{pie_html}')
    return _envolver(cuerpo_html, "diapositiva", "pf-diapo" + (" pf-dia-portada" if portada else ""),
                     hoja_extra="".join(hojas))


def cifra_grande(lista, titulo=None, decimales=0):
    """Una o varias cifras protagonistas ("héroe"), con conteo al aparecer.
    Cada elemento es un diccionario con `valor`, `etiqueta` y, opcionalmente,
    `detalle`, `color`, `decimales`, `prefijo`, `sufijo`; o una tupla
    `(valor, etiqueta[, detalle])`.
    Uso: `mostrar(cifra_grande([{"valor": 21.2, "etiqueta": "alertas por semana", "decimales": 1}]))`."""
    if isinstance(lista, dict):
        lista = [lista]
    piezas = []
    for i, elemento in enumerate(lista):
        campo = _campos(elemento, ("valor", "etiqueta", "detalle"))
        color = campo.get("color", "azul")
        cifra = _cifra(campo.get("valor"), campo.get("decimales", decimales),
                       str(campo.get("prefijo", "")), str(campo.get("sufijo", "")))
        detalle = (f'<div class="pf-heroe-det">{_texto(campo["detalle"])}</div>'
                   if campo.get("detalle") else "")
        piezas.append(
            f'<div class="pf-heroe pf-ap" style="--pf-i:{i * 2};--pf-c:{_color(color)};--pf-o:{_oscuro(color)}">'
            f'<div class="pf-heroe-valor">{cifra}</div>'
            f'<div class="pf-heroe-etq">{_texto(campo.get("etiqueta", ""))}</div>{detalle}</div>')
    return _envolver(f'<div class="pf-heroes">{"".join(piezas)}</div>', "heroe", titulo=titulo)


_FLECHA = ('<svg viewBox="0 0 24 14" aria-hidden="true"><path class="pf-trazo" pathLength="1" d="M1 7H21" '
           'stroke="#8C8C8C" stroke-width="2" fill="none"/><path d="M16 2.5L22 7L16 11.5" stroke="#8C8C8C" '
           'stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def flujo(pasos_flujo, titulo=None):
    """Proceso horizontal: cajas unidas por flechas que se dibujan en orden.
    Cada paso es `(titulo, detalle)` o un diccionario con `titulo`, `detalle`
    y, opcionalmente, `marca` (texto pequeño encima), `color` y `lleno`
    (caja rellena del color, para destacar).
    Uso: `mostrar(flujo([("Entrenamiento", "ajustar"), ("Validación", "decidir"), {"titulo": "Prueba", "detalle": "una vez", "color": "rojo", "lleno": True}]))`."""
    piezas = []
    for i, paso in enumerate(pasos_flujo):
        campo = _campos(paso, ("titulo", "detalle"))
        color = campo.get("color", "azul")
        if i:
            piezas.append(f'<div class="pf-flujo-flecha" style="--pf-i:{i - 1}">{_FLECHA}</div>')
        marca = f'<div class="pf-flujo-marca">{_texto(campo["marca"])}</div>' if campo.get("marca") else ""
        detalle = f'<div class="pf-flujo-det">{_texto(campo["detalle"])}</div>' if campo.get("detalle") else ""
        lleno = " pf-flujo-lleno" if campo.get("lleno") else ""
        fondo = _oscuro(color) if campo.get("lleno") else _color(color)
        piezas.append(
            f'<div class="pf-flujo-paso{lleno}" style="--pf-i:{i};--pf-c:{fondo};--pf-o:{_oscuro(color)}">'
            f'{marca}<div class="pf-flujo-titulo">{_texto(campo.get("titulo", ""))}</div>{detalle}</div>')
    return _envolver(f'<div class="pf-flujo">{"".join(piezas)}</div>', "flujo", titulo=titulo)


def matriz_confusion(vn, fp, fn, vp, titulo=None, clases=("Conforme", "No conforme"),
                     eje_real="Resultado real", eje_predicho="Alerta del modelo", predichos=("Sin alerta", "Alerta"),
                     rotulos=("bien descartadas", "alertas innecesarias", "no conformes sin alerta",
                              "no conformes detectadas"), pie=None):
    """Matriz de confusión 2 × 2 como cuadrícula de HTML: filas = clase real,
    columnas = decisión del modelo. Cada celda lleva su conteo y qué significa.
    Uso: `mostrar(matriz_confusion(7202, 680, 523, 252, "Prueba"))`."""
    celdas = list(zip((vn, fp, fn, vp), rotulos, COLORES_CONFUSION))
    html = [f'<div></div><div class="pf-mc-eje">{_texto(predichos[0])}</div>'
            f'<div class="pf-mc-eje">{_texto(predichos[1])}</div>']
    for fila in range(2):
        html.append(f'<div class="pf-mc-fila">{_texto(clases[fila])}</div>')
        for columna in range(2):
            valor, rotulo, color = celdas[2 * fila + columna]
            html.append(
                f'<div class="pf-mc-celda" style="--pf-i:{2 * fila + columna};--pf-c:{_oscuro(color)};'
                f'--pf-f:{_tenue(color)}"><div class="pf-mc-valor">{_cifra(valor, 0)}</div>'
                f'<div class="pf-mc-etq">{_texto(rotulo)}</div></div>')
    pie_html = (f'<div class="pf-mc-pie">{_texto(eje_real)} en filas y {_texto(eje_predicho).lower()} en columnas'
                + (f". {_texto(pie)}" if pie else "") + "</div>")
    return _envolver(f'<div class="pf-mc">{"".join(html)}</div>{pie_html}', "matriz", titulo=titulo)


def linea_svg(x, series, titulo=None, marca=None, rotulo_marca=None, etiqueta_x="", formato_y=2,
              ancho=560, alto=210):
    """Gráfico de líneas pequeño en SVG, sin bibliotecas: cada línea se
    dibuja de izquierda a derecha al aparecer. `series` es un diccionario
    `{nombre: valores}` o `{nombre: (valores, color)}`; `marca` es un valor
    de `x` que se señala con una línea vertical y un punto sobre la primera
    serie; `formato_y` da formato a las etiquetas del eje vertical.
    Uso: `mostrar(linea_svg(umbrales, {"Costo": costos}, marca=0.167, rotulo_marca="umbral elegido"))`."""
    x = [float(v) for v in x]
    pares = []
    for i, (nombre, contenido) in enumerate(_pares(series)):
        valores, color = (contenido if isinstance(contenido, tuple) else (contenido, None))
        pares.append((nombre, [float(v) for v in valores],
                      _color(color) if color else config.PALETA[i % len(config.PALETA)]))
    todos = [v for _, valores, _ in pares for v in valores if not _falta(v)]
    y_min, y_max = min(todos), max(todos)
    if y_max == y_min:
        y_max = y_min + 1
    margen_y = 0.08 * (y_max - y_min)
    y_min, y_max = y_min - margen_y, y_max + margen_y
    izq, der, arriba, abajo = 54, 14, 14, 34
    x_min, x_max = min(x), max(x)
    escala_x = (lambda v: izq + (ancho - izq - der) * (v - x_min) / ((x_max - x_min) or 1))
    escala_y = (lambda v: arriba + (alto - arriba - abajo) * (y_max - v) / (y_max - y_min))
    partes = [f'<line x1="{izq}" y1="{alto - abajo}" x2="{ancho - der}" y2="{alto - abajo}" stroke="{LINEA}"/>']
    for valor in (y_min + margen_y, (y_min + y_max) / 2, y_max - margen_y):
        y = escala_y(valor)
        partes.append(f'<line x1="{izq}" y1="{y:.1f}" x2="{ancho - der}" y2="{y:.1f}" stroke="{LINEA}" '
                      f'stroke-dasharray="3 4"/><text x="{izq - 8}" y="{y + 4:.1f}" font-size="12" '
                      f'text-anchor="end">{escape(_aplicar_formato(valor, formato_y))}</text>')
    enteros = all(v.is_integer() for v in x)
    marcas_x = sorted(set(x)) if enteros and len(set(x)) <= 12 else [x_min, x_max]
    for valor in marcas_x:
        ancla = "start" if valor == x_min and not enteros else "end" if valor == x_max and not enteros else "middle"
        texto = str(int(valor)) if enteros else formato_numero(valor, 2)
        partes.append(f'<text x="{escala_x(valor):.1f}" y="{alto - abajo + 18}" font-size="12" '
                      f'text-anchor="{ancla}">{escape(texto)}</text>')
    if etiqueta_x:
        partes.append(f'<text class="pf-lsvg-etq" x="{(izq + ancho - der) / 2:.1f}" y="{alto - 2}" font-size="12.5" '
                      f'text-anchor="middle">{escape(str(etiqueta_x))}</text>')
    for i, (_, valores, color) in enumerate(pares):
        puntos = " ".join(f"{escala_x(a):.1f},{escala_y(b):.1f}" for a, b in zip(x, valores) if not _falta(b))
        partes.append(f'<polyline class="pf-lsvg-serie pf-trazo" pathLength="1" points="{puntos}" '
                      f'stroke="{color}" style="--pf-i:{i * 3}"/>')
    if marca is not None and pares:
        cerca = min(range(len(x)), key=lambda k: abs(x[k] - float(marca)))
        mx, my = escala_x(x[cerca]), escala_y(pares[0][1][cerca])
        partes.append(f'<line x1="{mx:.1f}" y1="{arriba}" x2="{mx:.1f}" y2="{alto - abajo}" stroke="{COLORES["tinta"]}" '
                      f'stroke-dasharray="4 4" class="pf-ap" style="--pf-i:8"/>'
                      f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="6" fill="{PAPEL}" stroke="{pares[0][2]}" stroke-width="3" '
                      f'class="pf-ap" style="--pf-i:9"/>')
        if rotulo_marca:
            ancla = "start" if mx < (izq + ancho - der) / 2 else "end"
            desplazamiento = 8 if ancla == "start" else -8
            partes.append(f'<text x="{mx + desplazamiento:.1f}" y="{arriba + 12}" font-size="13" font-weight="650" '
                          f'text-anchor="{ancla}" class="pf-ap" style="--pf-i:9;fill:{COLORES["tinta"]}">'
                          f'{escape(str(rotulo_marca))}</text>')
    leyenda = ""
    if len(pares) > 1:
        leyenda = '<div class="pf-lsvg-leyenda">' + "".join(
            f'<span><span class="pf-lsvg-muestra" style="--pf-c:{c}"></span>{_texto(n)}</span>'
            for n, _, c in pares) + "</div>"
    svg = (f'<svg viewBox="0 0 {ancho} {alto}" role="img" aria-label="{escape(str(titulo or "Gráfico"), quote=True)}">'
           f'{"".join(partes)}</svg>')
    return _envolver(f'<div class="pf-lsvg">{svg}{leyenda}</div>', "linea_svg", titulo=titulo)


def columnas(*contenidos, proporciones=None, titulo=None):
    """Dos o más columnas lado a lado (una sola en pantallas estrechas).
    `proporciones` es una plantilla de CSS grid, por ejemplo `"3fr 2fr"`.
    Uso: `mostrar(columnas(barras_animadas(...), matriz_confusion(...), proporciones="1fr 1fr"))`."""
    plantilla = proporciones or " ".join(["1fr"] * len(contenidos))
    if not re.fullmatch(r"[0-9a-z.%()\s,]+", plantilla):
        raise ValueError(f"Proporciones no válidas: {plantilla!r}")
    cuerpo = "".join(f'<div class="pf-col">{_a_html(c)}</div>' for c in contenidos)
    return _envolver(f'<div class="pf-columnas" style="grid-template-columns:{plantilla}">{cuerpo}</div>',
                     "columnas", titulo=titulo)


def fichas(lista, titulo=None, color="azul"):
    """Etiquetas en forma de píldora que aparecen en secuencia (temas, variables).
    Uso: `mostrar(fichas(["KNN", "Regresión logística", "Árboles"], "Temas del curso"))`."""
    piezas = "".join(
        f'<span class="pf-ficha pf-ap" style="--pf-i:{i};--pf-c:{_color(color)};--pf-f:{_tenue(color)}">'
        f"{_texto(e)}</span>" for i, e in enumerate(lista))
    return _envolver(f'<div class="pf-fichas">{piezas}</div>', "fichas", titulo=titulo)


def fraccion(izquierda, numerador, denominador, detalle=None):
    """Fórmula con una fracción, en HTML. En numerador y denominador, `[[a:texto]]`
    se pinta en verde y `[[b:texto]]` en rojo (además de `**negrita**`).
    Uso: `mostrar(fraccion("VPP", "[[a:S p]]", "[[a:S p]] + [[b:(1 − E)(1 − p)]]"))`."""
    def pintar(texto):
        return re.sub(r"\[\[([ab]):(.+?)\]\]", r'<span class="pf-f-\1">\2</span>', _texto(texto))
    det = f'<div class="pf-f-det">{_texto(detalle)}</div>' if detalle else ""
    cuerpo = (f'<div class="pf-formula pf-ap"><span class="pf-f-izq">{_texto(izquierda)}</span><span>=</span>'
              f'<span class="pf-f-frac"><span class="pf-f-num">{pintar(numerador)}</span>'
              f'<span class="pf-f-den">{pintar(denominador)}</span></span>{det}</div>')
    return _envolver(cuerpo, "formula")


def recuadros(lista, titulo=None, por_fila=None):
    """Recuadros con una marca circular (número o símbolo), título y texto;
    sirven para preguntas, respuestas o pasos siguientes. Cada elemento es
    `(marca, titulo, texto)` o un diccionario con esas claves y `color`.
    Uso: `mostrar(recuadros([("1", "¿Cuántas muestras llegarán?", "Regresión"), ("2", "¿Cuáles son de riesgo?", "Clasificación")]))`."""
    piezas = []
    for i, elemento in enumerate(lista):
        campo = _campos(elemento, ("marca", "titulo", "texto"))
        color = _oscuro(campo.get("color", "azul"))
        texto = f'<div class="pf-rec-texto">{_texto(campo["texto"])}</div>' if campo.get("texto") else ""
        piezas.append(
            f'<div class="pf-recuadro pf-ap" style="--pf-i:{i * 2};--pf-c:{color}">'
            f'<div class="pf-rec-marca">{_texto(campo.get("marca", i + 1))}</div>'
            f'<div><div class="pf-rec-titulo">{_texto(campo.get("titulo", ""))}</div>{texto}</div></div>')
    n = int(por_fila or len(piezas) or 1)
    return _envolver(f'<div class="pf-recuadros" style="grid-template-columns:repeat({n},minmax(0,1fr))">'
                     f'{"".join(piezas)}</div>', "recuadros", titulo=titulo)
