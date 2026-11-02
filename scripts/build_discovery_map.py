"""Compose the author's original article/PPT artwork into a responsive workflow.

Only original picture cropping, scaling and composition are performed. Coloured
layout frames, article terms and straight connectors remain editable SVG. The
packaged cropped assets allow rebuilding without the external source PowerPoint.
"""
from pathlib import Path
from html import escape
from io import BytesIO
import argparse
import base64
import hashlib
import json
from zipfile import ZipFile
from PIL import Image, ImageChops

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'docs/assets/workflow'
# Original PPT media, never redrawn or retouched. Whitespace-only trims preserve
# source pixels. The output summary omits the old numeric regime-name column.
PICTURES={
    'description-original':('image9.png',None),
    'dataset-original':('image10.png',None),
    'input-link-original':('image11.png',None),
    'grounding-original':('image6.png',None),
    'basis-original':('image8.png',None),
    'planning-original':('image7.png',None),
    'regimes-original':('image1.png',None),
    'similarity-original':('image2.png',None),
    'mi-bars-original':('image3.png',None),
    'sparsification-original':('image4.png',None),
    'rationalization-original':('image5.png',None),
    'gpr-validation-original':('image19.png',None),
    'scaling-summary-original':('image12.png',(412,168,1856,622)),
    'regime-collapse-original':('image13.png',None),
    'regime-summary-original':('image14.png',None),
}


def extract_originals(source):
    deck=source/'figs_revised_2610/Fig2_workflow_.pptx'
    manifest=ART/'original-artwork-provenance.json'
    if not deck.exists():
        missing=[name for name in PICTURES if not (ART/(name+'.png')).exists()]
        if missing:raise FileNotFoundError('Original artwork is missing: '+', '.join(missing))
        return
    records=[]
    with ZipFile(deck) as archive:
        for name,(media,explicit_crop) in PICTURES.items():
            data=archive.read('ppt/media/'+media)
            original=Image.open(BytesIO(data)).convert('RGBA')
            if explicit_crop:box=explicit_crop
            else:
                white=Image.new('RGBA',original.size,'white');white.alpha_composite(original)
                difference=ImageChops.difference(white.convert('RGB'),Image.new('RGB',original.size,'white'))
                box=difference.point(lambda v:255 if v>22 else 0).getbbox() or (0,0,*original.size)
                box=(max(0,box[0]-3),max(0,box[1]-3),min(original.width,box[2]+3),min(original.height,box[3]+3))
            target=ART/(name+'.png')
            original.crop(box).save(target,optimize=True)
            records.append(dict(asset=target.relative_to(ROOT).as_posix(),source='figs_revised_2610/Fig2_workflow_.pptx',source_sha256=hashlib.sha256(deck.read_bytes()).hexdigest(),part='ppt/media/'+media,original_image_sha256=hashlib.sha256(data).hexdigest(),crop_box=list(box),method='Crop original pixels only; omit surrounding whitespace.' if not explicit_crop else 'Crop original summary to its group-count, similarity, MI, R² and formulation columns; omit numeric regime-name column.'))
    manifest.write_text(json.dumps({'illustrations':records,'policy':'Original author-supplied picture media only. No diagram redrawing, recoloring, inpainting or invented data.'},indent=2),encoding='utf-8')


def label(x,y,value,size=18,**attrs):
    attributes=' '.join(f'{k.replace("_","-")}="{v}"' for k,v in attrs.items())
    return f'<text x="{x}" y="{y}" font-size="{size}" {attributes}>{escape(value)}</text>'


def rect(x,y,w,h,fill,stroke='none',radius=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'


def image(name,x,y,w,h):
    data=base64.b64encode((ART/(name+'.png')).read_bytes()).decode()
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{data}"/>'


def arrow(x1,y1,x2,y2):
    return f'<path d="M{x1} {y1}L{x2} {y2}" fill="none" stroke="#597780" stroke-width="1.6" marker-end="url(#arrow)"/>'


def build(source):
    ART.mkdir(parents=True,exist_ok=True)
    extract_originals(source)
    pieces=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 990" role="img" aria-label="AI Buckingham: original article artwork for language/data inputs, semantic planning, numerical inference and regime-wise reporting"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="none" stroke="#597780" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs><rect width="1200" height="990" fill="white"/><g font-family="Arial,sans-serif" fill="#29464d">']
    def layer(y,h,title,fill):
        pieces.append(rect(20,y,1160,h,fill,radius=12)+label(40,y+29,title,21,font_weight='600'))
    def card(x,y,w,h,title,fill):
        pieces.append(rect(x,y,w,h,fill,'white'))
        for i,value in enumerate(title.split('|')):
            pieces.append(label(x+16,y+28+i*22,value,20 if w>400 else 18.5,font_weight='600'))
    # Source-original description, file and link icons.
    pieces.append(rect(20,14,560,68,'#eff6f2','#d7e5dc'))
    pieces.append(image('description-original',32,22,56,52))
    pieces.append(label(105,43,'Natural-language description',20,font_weight='600'))
    pieces.append(label(105,65,'Physical context and governing variables',15,fill='#637c76'))
    pieces.append(image('input-link-original',585,29,30,30))
    pieces.append(rect(620,14,560,68,'#eff6f2','#d7e5dc'))
    pieces.append(image('dataset-original',635,22,56,52))
    pieces.append(label(708,43,'User-provided dataset file',20,font_weight='600'))
    pieces.append(label(708,65,'Observations and dimensionless response',15,fill='#637c76'))
    pieces.append(arrow(600,87,600,104))

    # Semantic modules: trim whitespace before placing the original picture.
    layer(110,208,'LLM semantic-planning layer','#e5effb')
    card(40,152,350,152,'Semantic parsing','#d9eafb')
    pieces.append(image('grounding-original',50,188,330,89))
    pieces.append(label(57,293,'Dimensional matrix · physical consistency',14,fill='#536f72'))
    card(425,152,350,152,'Basis construction','#e9e1f6')
    pieces.append(image('basis-original',435,188,330,89))
    pieces.append(label(441,293,'Null space · Buckingham–Π basis',14,fill='#536f72'))
    card(810,152,350,152,'Workflow planning','#f9eacb')
    pieces.append(image('planning-original',820,188,330,89))
    pieces.append(label(826,293,'Analysis route · route-specific settings',14,fill='#536f72'))

    # Numerical modules use the full source groups, not a single partial image.
    layer(347,354,'Numerical inference layer','#e8f3ed')
    card(40,389,350,298,'Gradient-based physical|regime partition','#ccebdc')
    pieces.append(image('regimes-original',51,443,328,226))
    card(425,389,350,298,'MI-based dimensionless|learning','#d2eaf3')
    pieces.append(image('similarity-original',440,446,320,139))
    pieces.append(image('mi-bars-original',440,590,320,94))
    card(810,389,350,298,'Formulations refinement|and evaluation','#f5dfd9')
    pieces.append(label(828,454,'Sparsification',14,fill='#536f72'))
    pieces.append(label(996,454,'Rationalization',14,fill='#536f72'))
    pieces.append(image('sparsification-original',822,461,154,112))
    pieces.append(image('rationalization-original',991,461,154,112))
    pieces.append(label(827,590,'Held-out GPR validation',15,fill='#536f72'))
    pieces.append(image('gpr-validation-original',822,601,326,77))

    # Original synthesis summary and original collapse/fraction report.
    layer(730,241,'Synthesis and reporting layer','#efe9f6')
    card(40,772,540,185,'Discovered dimensionless scaling','#e1d5f0')
    pieces.append(image('scaling-summary-original',52,814,516,129))
    card(620,772,540,185,'Regime-wise visualization report','#dbeaf2')
    pieces.append(image('regime-collapse-original',632,813,363,133))
    pieces.append(image('regime-summary-original',1004,813,143,133))

    for y in [228,537]:
        for x in [393,778]:pieces.append(arrow(x,y,x+29,y))
    for y in [322,705]:pieces.append(arrow(600,y,600,y+20))
    pieces.append(arrow(583,865,616,865))
    pieces.append('</g></svg>')
    (ROOT/'docs/assets/discovery-map.svg').write_text(''.join(pieces),encoding='utf-8')
    print(f'Built workflow from {len(PICTURES)} original PPT picture crops.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT.parent)
    build(parser.parse_args().source.resolve())
