"""Search actual Commons images and download bounded, rasterized previews."""
from io import BytesIO
import html
import json
import re
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen, HTTPRedirectHandler, build_opener
from PIL import Image, ImageOps

USER_AGENT='HTNProjector/0.1 (https://github.com/Johannestampere/HackTheNorth2026)'
HOSTS={'commons.wikimedia.org','upload.wikimedia.org','thumb.wikimedia.org'}


def check_url(url):
    parsed=urlsplit(url)
    if parsed.scheme!='https' or parsed.hostname not in HOSTS or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Expected a Wikimedia HTTPS image URL')


class WikimediaRedirects(HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        check_url(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def fetch(url,limit):
    check_url(url)
    opener=build_opener(WikimediaRedirects())
    with opener.open(Request(url,headers={'User-Agent':USER_AGENT}),timeout=20) as response:
        data=response.read(limit+1)
    if len(data)>limit: raise ValueError('Image/search response exceeds download limit')
    return data


def plain(value):
    return html.unescape(re.sub('<[^>]+>','',value or '')).strip()


class WebImages:
    """Commons search, source metadata and bounded downloads; no generated URLs."""
    def search(self,query):
        if not isinstance(query,str) or not 1<=len(query.strip())<=150:
            raise ValueError('Image query must contain 1–150 characters')
        original=query.strip()
        words=original.split()
        queries=list(dict.fromkeys([original,' '.join(words[:4]),' '.join(words[:2])]))
        for simplified in queries:
            candidates=self._search_once(simplified)
            if candidates:
                for candidate in candidates: candidate['requested_query']=original
                return candidates
        raise RuntimeError('No usable image found after broader searches')

    def _search_once(self,query):
        params=dict(action='query',format='json',generator='search',gsrsearch=query,
                    gsrnamespace=6,gsrlimit=8,prop='imageinfo',
                    iiprop='url|mime|size|extmetadata',iiurlwidth=1600)
        result=json.loads(fetch('https://commons.wikimedia.org/w/api.php?'+urlencode(params),2*1024*1024))
        if 'error' in result: raise RuntimeError('Commons rejected the search')
        candidates=[]
        for page in sorted(result.get('query',{}).get('pages',{}).values(),key=lambda p:p.get('index',999)):
            info=page.get('imageinfo',[{}])[0]
            if info.get('mime') not in ('image/jpeg','image/png','image/webp','image/svg+xml'): continue
            url=info.get('thumburl') or info.get('url')
            if not url or (info.get('mime')=='image/svg+xml' and not info.get('thumburl')): continue
            width=info.get('thumbwidth',info.get('width',0)); height=info.get('thumbheight',info.get('height',0))
            if min(width,height)<240: continue
            check_url(url)
            meta=info.get('extmetadata',{})
            def field(key): return plain(meta.get(key,{}).get('value',''))
            candidates.append(dict(title=page['title'],image_url=url,source_url=info.get('descriptionurl',''),
                width=width,height=height,description=field('ImageDescription')[:700],
                author=field('Artist'),credit=field('Credit'),license=field('LicenseShortName'),
                license_url=field('LicenseUrl'),attribution=field('Attribution'),query=query))
        return candidates

    def download(self,candidate):
        data=fetch(candidate['image_url'],12*1024*1024)
        with Image.open(BytesIO(data)) as source:
            if source.width*source.height>20_000_000: raise ValueError('Image dimensions exceed limit')
            source=ImageOps.exif_transpose(source)
            rgba=source.convert('RGBA')
            # Transparent diagram backgrounds generally need white to retain black labels.
            image=Image.new('RGBA',rgba.size,'white'); image.alpha_composite(rgba)
            return image.convert('RGB')


def fullscreen(image,width,height):
    """Preserve the complete image and its labels; letterbox, never crop or distort."""
    frame=Image.new('RGB',(width,height),'black')
    fitted=ImageOps.contain(image.convert('RGB'),(width,height),Image.Resampling.LANCZOS)
    frame.paste(fitted,((width-fitted.width)//2,(height-fitted.height)//2))
    return frame
