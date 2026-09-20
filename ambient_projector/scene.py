"""Validated visual plan shared by the AI and deterministic drawing code."""
from dataclasses import dataclass, asdict
from math import isfinite

COLORS = {'white': '#f5f5f5', 'cyan': '#65dcff', 'yellow': '#ffe07a',
          'green': '#8befa3', 'pink': '#ff9aca'}
KINDS = ('text', 'line', 'arrow', 'rectangle', 'ellipse')

# All element fields are required for strict structured output. For text, x/y
# and x2/y2 delimit the text box; for shapes they are corners or endpoints.
ELEMENT_SCHEMA = {'type': 'object', 'additionalProperties': False,
    'properties': {'kind': {'type': 'string', 'enum': list(KINDS)},
                   **{k: {'type': 'number'} for k in ('x','y','x2','y2')},
                   'text': {'type': 'string'},
                   'color': {'type': 'string', 'enum': list(COLORS)}},
    'required': ['kind','x','y','x2','y2','text','color']}
SCHEMA = {'type': 'object', 'additionalProperties': False,
    'properties': {'title': {'type': 'string'},
                   'notes': {'type': 'array', 'items': {'type': 'string'}},
                   'elements': {'type': 'array', 'items': ELEMENT_SCHEMA},
                   'illustration_prompt': {'type': 'string'}},
    'required': ['title','notes','elements','illustration_prompt']}


def text_value(value, limit):
    if not isinstance(value,str) or len(value)>limit or (value and not value.isprintable()):
        raise ValueError(f'Text must be printable and at most {limit} characters')
    return value


@dataclass(frozen=True)
class Element:
    """One diagram primitive in a normalized 0–1 canvas, not camera coordinates."""
    kind: str
    x: float
    y: float
    x2: float
    y2: float
    text: str
    color: str

    def __post_init__(self):
        if self.kind not in KINDS or self.color not in COLORS:
            raise ValueError('Unsupported drawing primitive or color')
        for v in (self.x,self.y,self.x2,self.y2):
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not isfinite(v) or not 0<=v<=1:
                raise ValueError('Drawing coordinates must be finite values in [0,1]')
        text_value(self.text,100)
        if self.kind in ('text','rectangle','ellipse') and (self.x2<=self.x or self.y2<=self.y):
            raise ValueError('A box needs positive width and height')
        if self.kind=='text' and not self.text.strip():
            raise ValueError('Text elements need text')


@dataclass(frozen=True)
class Scene:
    """A short heading, up to three notes, a diagram and optional illustration brief.

    The diagram remains usable if image generation is disabled or fails.
    The generated illustration, when available, replaces the diagram only.
    """
    title: str
    notes: tuple[str,...]
    elements: tuple[Element,...]
    illustration_prompt: str

    def __post_init__(self):
        text_value(self.title,70)
        if not self.title.strip() or len(self.notes)>3 or len(self.elements)>32:
            raise ValueError('Scene needs a title, at most 3 notes and 32 drawing elements')
        for note in self.notes: text_value(note,160)
        text_value(self.illustration_prompt,1500)

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!=set(SCHEMA['required']):
            raise ValueError('Unexpected visual-plan fields')
        if not isinstance(data['notes'],list) or not isinstance(data['elements'],list):
            raise ValueError('Notes and elements must be arrays')
        return cls(data['title'],tuple(data['notes']),tuple(Element(**e) for e in data['elements']),data['illustration_prompt'])

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TextPage:
    """Flexible plain text: a title and body, with paragraph/list structure intact.

    Length limits bound input size only; there is no required number of sections,
    bullets or paragraphs. The renderer independently enforces readable fitting.
    """
    title: str
    body: str

    def __post_init__(self):
        text_value(self.title,200)
        if not self.title.strip() or not isinstance(self.body,str) or not self.body.strip() or len(self.body)>4000:
            raise ValueError('Text page needs a title and a nonempty body of at most 4000 characters')
        if not self.body.replace('\n','').isprintable():
            raise ValueError('Body may contain printable text and newlines only')

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)!={'title','body'}:
            raise ValueError('Text page must contain title and body')
        return cls(**data)

    def as_dict(self): return asdict(self)
