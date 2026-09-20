"""Render validated text, diagrams and optional illustrations into RGB pixels."""
from math import atan2, cos, sin, pi
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .scene import COLORS


def fit_text(draw,text,box,color,max_size):
    """Wrap text, shrinking within a readable range; reject rather than clip."""
    # Pillow's bundled font lacks several common typographic punctuation marks.
    text=text.translate(str.maketrans({'—':' - ','–':'-','‘':"'",'’':"'",'“':'"','”':'"','→':' -> ','…':'...'}))
    x,y,right,bottom=box
    for size in range(max_size,11,-1):
        font=ImageFont.load_default(size=size)
        lines=[]; line=''
        for word in text.split():
            proposed=(line+' '+word).strip()
            if draw.textlength(proposed,font=font)<=right-x:
                line=proposed
            else:
                if line: lines.append(line)
                line=''
                for char in word:
                    if draw.textlength(line+char,font=font)>right-x:
                        if line: lines.append(line)
                        line=''
                    line+=char
        if line: lines.append(line)
        step=int(size*1.35)
        if len(lines)*step<=bottom-y and all(draw.textlength(s,font=font)<=right-x for s in lines):
            for line in lines:
                draw.text((x,y),line,font=font,fill=color,anchor='lt'); y+=step
            return
    raise ValueError('Text does not fit its box; model needs a simpler layout')


class Renderer:
    """Fixed wall layout with heading above a diagram and short notes below."""
    def __init__(self,width=1280,height=720):
        if width<640 or height<480 or width*height*3>64*1024*1024:
            raise ValueError('Use at least 640×480 and at most 64 MiB of RGB pixels')
        self.width,self.height=width,height

    def render(self,scene,illustration=None):
        w,h=self.width,self.height
        image=Image.new('RGB',(w,h),'black'); draw=ImageDraw.Draw(image)
        margin=int(min(w,h)*.04)
        fit_text(draw,scene.title,(margin,margin,w-margin,int(h*.15)),COLORS['cyan'],max(20,int(h*.065)))
        left,top,right,bottom=margin,int(h*.18),w-margin,int(h*.70)
        # Draw into a separate canvas so arrows and shapes cannot cross into text.
        diagram=Image.new('RGB',(right-left,bottom-top),'black')
        if illustration is not None:
            contained=ImageOps.contain(illustration,diagram.size)
            diagram.paste(contained,((diagram.width-contained.width)//2,(diagram.height-contained.height)//2))
        else:
            pen=ImageDraw.Draw(diagram)
            stroke=max(2,int(h/240))
            for e in scene.elements:
                a=(e.x*(diagram.width-1),e.y*(diagram.height-1))
                b=(e.x2*(diagram.width-1),e.y2*(diagram.height-1))
                color=COLORS[e.color]
                if e.kind=='text':
                    fit_text(pen,e.text,(*a,*b),color,max(18,int(h*.038)))
                elif e.kind in ('line','arrow'):
                    pen.line((a,b),fill=color,width=stroke)
                    if e.kind=='arrow' and a!=b:
                        angle=atan2(b[1]-a[1],b[0]-a[0]); length=12*h/720
                        points=[b]+[(b[0]-length*cos(angle+d),b[1]-length*sin(angle+d)) for d in (-pi/6,pi/6)]
                        pen.polygon(points,fill=color)
                elif e.kind=='rectangle': pen.rectangle((*a,*b),outline=color,width=stroke)
                elif e.kind=='ellipse': pen.ellipse((*a,*b),outline=color,width=stroke)
        image.paste(diagram,(left,top))
        for i,note in enumerate(scene.notes):
            y=int(h*.74)+i*int(h*.075)
            fit_text(draw,note,(margin,y,w-margin,y+int(h*.07)),COLORS['white'],max(18,int(h*.033)))
        if illustration is not None:
            draw.text((margin,h-18),'AI-generated illustration',font=ImageFont.load_default(size=12),fill='#aaaaaa')
        return image
