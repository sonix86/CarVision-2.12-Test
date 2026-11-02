"""Local, session-scoped phone uploads. No model or cloud service in this module."""
import io
import json
import secrets
import socket
import threading
import warnings
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from PIL import Image, ImageOps

MAX_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 40_000_000


def local_addresses():
    """Enumerate usable local IPv4 addresses without contacting the internet."""
    addresses = set()
    preferred = []
    try:
        addresses.update(x[4][0] for x in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    # UDP connect chooses an interface; no data is sent.
    for network in ('192.168.1.1', '192.168.42.1', '172.20.10.1'):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect((network, 9))
                address = sock.getsockname()[0]
                addresses.add(address)
                if address not in preferred:
                    preferred.append(address)
        except OSError:
            pass
    usable = {x for x in addresses if x != '0.0.0.0' and not x.startswith('127.')}
    return [x for x in preferred if x in usable] + sorted(usable - set(preferred))


class PhoneReceiver:
    """One outstanding upload, bounded history, and a callback to the GUI thread."""
    def __init__(self, directory, submit, host='0.0.0.0', port=8765):
        Image.init()  # Discover image plugins on the caller/main thread, not in a request.
        self.directory = Path(directory) / secrets.token_hex(8)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.submit = submit
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.RLock()
        self.jobs = OrderedDict()
        self.active = None
        self.stopped = False
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.0'
            def setup(self):
                super().setup()
                self.connection.settimeout(30)

            def log_message(self, *_):
                pass  # Do not log the session URL/token or phone addresses.

            def reply(self, code, data, content_type='application/json; charset=utf-8'):
                if isinstance(data, dict):
                    data = json.dumps(data, ensure_ascii=False).encode('utf-8')
                elif isinstance(data, str):
                    data = data.encode('utf-8')
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; media-src blob:; frame-ancestors 'none'")
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (OSError, TimeoutError):
                    pass

            def route(self):
                path = urlsplit(self.path).path
                parts = path.split('/')
                if len(parts) < 3 or not secrets.compare_digest(parts[1], receiver.token):
                    self.reply(404, {'error': 'Ссылка недействительна. Откройте адрес из приложения.'})
                    return None
                return '/'.join(parts[2:])

            def do_GET(self):
                route = self.route()
                if route is None:
                    return
                if route == '':
                    self.reply(200, PAGE, 'text/html; charset=utf-8')
                    return
                if route.startswith('jobs/'):
                    parts = route.split('/')
                    with receiver.lock:
                        job = receiver.jobs.get(parts[1])
                        if job is None:
                            self.reply(404, {'error': 'Задание не найдено.'})
                        elif len(parts) == 2:
                            self.reply(200, {k: v for k, v in job.items() if k not in ('path', 'preview')})
                        elif parts[2:] == ['preview'] and job.get('preview'):
                            self.reply(200, Path(job['preview']).read_bytes(), 'image/jpeg')
                        else:
                            self.reply(404, {'error': 'Результат ещё не готов.'})
                    return
                self.reply(404, {'error': 'Not found'})

            def do_POST(self):
                route = self.route()
                if route is None:
                    return
                if route != 'upload':
                    self.reply(404, {'error': 'Not found'})
                    return
                # Same-origin phone page only. Never enable permissive CORS.
                origin = self.headers.get('Origin')
                if origin and origin != 'http://' + self.headers.get('Host', ''):
                    self.reply(403, {'error': 'Откройте страницу телефона из приложения.'})
                    return
                try:
                    if self.headers.get('Transfer-Encoding'):
                        raise ValueError('Неподдерживаемый способ передачи.')
                    size = int(self.headers.get('Content-Length', '0'))
                    if not 0 < size <= MAX_BYTES:
                        self.reply(413, {'error': 'Нужно фото размером до 15 МБ.'})
                        return
                    ident = receiver.reserve()
                    if ident is None:
                        self.reply(409, {'error': 'Предыдущее фото ещё обрабатывается. Подождите.'})
                        return
                    try:
                        raw = self.rfile.read(size)
                        if len(raw) != size:
                            raise ValueError('Фото передано не полностью. Отправьте ещё раз.')
                        path = receiver.normalise(raw, ident)
                        with receiver.lock:
                            if receiver.stopped:
                                raise ValueError('Приём остановлен на компьютере.')
                            receiver.jobs[ident]['path'] = path
                        receiver.submit(ident, path)
                    except Exception:
                        receiver.fail(ident, 'Не удалось принять фото. Попробуйте JPEG, PNG или WebP.')
                        raise
                    self.reply(202, {'id': ident})
                except (ValueError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
                    self.reply(400, {'error': str(exc) if isinstance(exc, ValueError) else 'Фото не открывается. Выберите JPEG, PNG или WebP.'})
                except Exception:
                    self.reply(500, {'error': 'Ошибка приёма. Проверьте приложение на компьютере.'})

        self.http = ThreadingHTTPServer((host, port), Handler)
        self.http.daemon_threads = True
        self.port = self.http.server_address[1]
        self.thread = threading.Thread(target=self.http.serve_forever, name='carvision-phone', daemon=True)
        self.thread.start()

    @property
    def urls(self):
        return [f'http://{ip}:{self.port}/{self.token}/' for ip in local_addresses()]

    @property
    def usb_url(self):
        return f'http://localhost:{self.port}/{self.token}/'

    def reserve(self):
        with self.lock:
            if self.active or self.stopped:
                return None
            while len(self.jobs) >= 20:
                _, old = self.jobs.popitem(last=False)
                # Keep full analysis exports. Limit temporary uploads/previews.
                for key in ('path', 'preview'):
                    if old.get(key):
                        Path(old[key]).unlink(missing_ok=True)
            ident = secrets.token_hex(16)
            self.jobs[ident] = {'id': ident, 'status': 'processing'}
            self.active = ident
            return ident

    def normalise(self, raw, ident):
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in ('JPEG', 'PNG', 'WEBP', 'BMP'):
                    raise ValueError('Выберите JPEG, PNG или WebP. Для HEIC включите JPEG на телефоне.')
                if source.width * source.height > MAX_PIXELS or min(source.size) < 20:
                    raise ValueError('Фото слишком большое или слишком маленькое.')
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((2560, 2560))
                path = self.directory / (ident + '.jpg')
                image.save(path, quality=94)
                return str(path)

    def complete(self, ident, result, preview=None):
        with self.lock:
            if self.stopped or ident not in self.jobs:
                return
            cars = []
            for row in result['cars']:
                cars.append({k: row[k] for k in ('id', 'brand', 'colour', 'model', 'brand_confidence', 'colour_reliable') if k in row})
            job = self.jobs[ident]
            job.update(status='done', cars=cars, seconds=result['seconds'], warnings=result.get('warnings', []))
            if preview:
                # A dedicated copy has bounded session retention; original exports remain.
                path = self.directory / (ident + '_preview.jpg')
                path.write_bytes(Path(preview).read_bytes())
                job['preview'] = str(path)
                job['preview_url'] = f'jobs/{ident}/preview'
            if self.active == ident:
                self.active = None

    def fail(self, ident, error):
        with self.lock:
            if ident in self.jobs:
                self.jobs[ident].update(status='error', error=str(error))
            if self.active == ident:
                self.active = None

    def stop(self):
        if self.stopped:
            return
        with self.lock:
            self.stopped = True
            if self.active:
                self.fail(self.active, 'Приём остановлен на компьютере.')
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(timeout=2)


def serve(args):
    """CLI alternative using exactly the same receiver and offline pipeline."""
    from .core import Pipeline, save_prediction
    output = Path(args.output)
    pipeline = None
    def submit(ident, path):
        def work():
            nonlocal pipeline
            try:
                if pipeline is None:
                    pipeline = Pipeline(args.models, mode='fast')
                result = pipeline.predict(path)
                preview = save_prediction(path, result, output/'phone_results'/ident)
                receiver.complete(ident, result, preview)
            except Exception as exc:
                receiver.fail(ident, str(exc))
        threading.Thread(target=work, daemon=True).start()
    receiver = PhoneReceiver(output/'phone_inbox', submit, host=args.host, port=args.port)
    ready = dict(urls=receiver.urls, usb_url=receiver.usb_url, port=receiver.port)
    if args.ready_file:
        target = Path(args.ready_file); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ready, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(ready, ensure_ascii=False), flush=True)
    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()


PAGE = r'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>CarVision — телефон</title>
<style>[hidden]{display:none!important}body{font:17px system-ui;max-width:720px;margin:auto;padding:20px;background:#f4f7fb;color:#203249}button,label.action{display:inline-block;background:#087f8c;color:white;border:0;border-radius:7px;padding:14px;margin:5px 0;font:inherit;cursor:pointer}input{display:none}button:disabled{opacity:.5}img,video{width:100%;border-radius:8px}#status{white-space:pre-wrap}td,th{text-align:left;padding:7px;border-bottom:1px solid #ccd}table{width:100%}small{display:block;margin:8px 0}</style>
<h1>CarVision</h1><p>Фото обрабатывает ваш компьютер. Интернет и установка приложения на телефон не нужны.</p>
<label class="action">Выбрать фото<input id="gallery" type="file" accept="image/*"></label>
<label class="action">Снять камерой<input id="camera" type="file" accept="image/*" capture="environment"></label>
<small>Выбранный или снятый кадр сразу отправится на анализ.</small>
<button id="startLive">Камера в браузере</button><button id="stopLive" hidden>Остановить камеру</button>
<small>Непрерывная камера работает в защищённом контексте: например, localhost через Android USB/ADB. Через обычный Wi-Fi используйте «Снять камерой».</small>
<video id="video" muted playsinline hidden></video><p id="status" role="status">Готово к отправке.</p>
<table id="results"></table><img id="preview" hidden alt="Результат анализа с номерами машин">
<small>Unknown означает, что марка не установлена. Точность такая же, как в настольной версии.</small>
<script>
const $=id=>document.getElementById(id);let busy=false,stream=null,live=false;
function status(t){$('status').textContent=t;}
async function jpeg(file){
 const url=URL.createObjectURL(file);try{const im=new Image();im.src=url;await im.decode();
 const k=Math.min(1,2560/Math.max(im.width,im.height));const c=document.createElement('canvas');c.width=Math.round(im.width*k);c.height=Math.round(im.height*k);const g=c.getContext('2d');g.fillStyle='white';g.fillRect(0,0,c.width,c.height);g.drawImage(im,0,0,c.width,c.height);
 return await new Promise(r=>c.toBlob(r,'image/jpeg',.94));}finally{URL.revokeObjectURL(url);}}
async function send(blob){
 if(busy){status('Дождитесь предыдущего результата.');return;}busy=true;
 try{status('Отправка фото…');const r=await fetch('upload',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob});const data=await r.json();if(!r.ok)throw Error(data.error);
 status('Анализ на компьютере… Первый кадр может занять больше времени. Не закрывайте страницу.');
 let job;for(;;){await new Promise(r=>setTimeout(r,900));const res=await fetch('jobs/'+data.id);job=await res.json();if(!res.ok)throw Error(job.error);if(job.status==='error')throw Error(job.error);if(job.status==='done')break;}
 $('results').replaceChildren();const head=$('results').insertRow();for(const t of ['№','Марка / модель','Цвет']){const c=document.createElement('th');c.textContent=t;head.appendChild(c);}
 for(const car of job.cars){const row=$('results').insertRow();for(const t of [car.id,car.brand+(car.brand!=='unknown'&&car.model&&car.model!=='unknown'?' / '+car.model:''),car.colour+(car.colour_reliable===false?' (не уверен)':'')])row.insertCell().textContent=t;}
 status('Готово. Найдено машин: '+job.cars.length+'. Анализ: '+job.seconds+' с.');
 if(job.preview_url){$('preview').src=job.preview_url;$('preview').hidden=false;}
 }catch(e){status('Не удалось завершить: '+e.message+'\nПроверьте подключение и приложение. Можно отправить фото ещё раз.');if(live)stop();}finally{busy=false;}
}
for(const id of ['gallery','camera'])$(id).onchange=async e=>{const f=e.target.files[0];if(!f)return;if(busy){status('Дождитесь предыдущего анализа.');e.target.value='';return;}if(live)stop();try{status('Подготовка фото…');await send(await jpeg(f));}catch(e){status('Фото не открывается. Попробуйте JPEG или PNG.');}e.target.value='';};
function stop(){live=false;if(stream)stream.getTracks().forEach(t=>t.stop());stream=null;$('video').hidden=true;$('stopLive').hidden=true;$('startLive').hidden=false;}
async function frames(){while(live){if(!busy&&$('video').videoWidth){const v=$('video'),c=document.createElement('canvas');const k=Math.min(1,1600/v.videoWidth);c.width=Math.round(v.videoWidth*k);c.height=Math.round(v.videoHeight*k);c.getContext('2d').drawImage(v,0,0,c.width,c.height);await send(await new Promise(r=>c.toBlob(r,'image/jpeg',.92)));}await new Promise(r=>setTimeout(r,2000));}}
$('startLive').onclick=async()=>{if(busy)return;try{if(!navigator.mediaDevices||!window.isSecureContext)throw Error('Используйте кнопку «Снять камерой» через Wi-Fi, либо localhost через Android USB/ADB.');stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});$('video').srcObject=stream;$('video').hidden=false;await $('video').play();live=true;$('stopLive').hidden=false;$('startLive').hidden=true;frames();}catch(e){stop();status(e.message);}};
$('stopLive').onclick=stop;window.addEventListener('pagehide',stop);
</script></html>'''
