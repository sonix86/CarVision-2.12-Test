"""Create a sourced, printable city-photo test booklet, not the exam report."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
temp=ROOT/'tmp/pdfs/photo-book';temp.mkdir(parents=True,exist_ok=True)
samples=ROOT/'samples';records=json.loads((samples/'sources.json').read_text(encoding='utf-8'))
output=ROOT/'output/pdf/Tashkent_Photo_Examples.pdf';output.parent.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Arial',r'C:\Windows\Fonts\arial.ttf'));pdfmetrics.registerFont(TTFont('ArialBold',r'C:\Windows\Fonts\arialbd.ttf'));pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='ArialBold',italic='Arial',boldItalic='ArialBold')
W,H=landscape(A4);c=canvas.Canvas(str(output),pagesize=(W,H));c.setTitle('Ташкент: фотографии для проверки CarVision');c.setAuthor('CarVision · София Рахманкулова (владелец проекта; авторы фото указаны отдельно)')
styles={name:ParagraphStyle(name,fontName='Arial',fontSize=size,leading=size*1.42,textColor=colour) for name,size,colour in [('body',11,'#34445a'),('small',8,'#536579')]}
def text(s,x,y,width,style='body'):
    p=Paragraph(s,styles[style]);_,height=p.wrap(width,H);p.drawOn(c,x,y-height);return height
def page(number,kicker,title):
    c.setFillColorRGB(.96,.97,.985);c.rect(0,0,W,H,fill=1,stroke=0)
    c.setFillColorRGB(.055,.14,.23);c.rect(0,H-100,W,100,fill=1,stroke=0)
    c.setFillColorRGB(.25,.83,.75);c.setFont('ArialBold',9);c.drawString(35,H-28,kicker)
    c.setFillColorRGB(1,1,1);c.setFont('ArialBold',23);c.drawString(35,H-65,title)
    c.setFillColorRGB(.4,.47,.56);c.setFont('Arial',8);c.drawString(35,20,'CARVISION / София Рахманкулова · примеры для проверки, не итоговый экзаменационный отчёт');c.drawRightString(W-35,20,f'{number:02d}')

page(1,'TASHKENT / PHOTO TEST KIT','Фотографии для проверки приложения')
text('Шесть реальных фотографий автомобилей в Ташкенте из Wikimedia Commons. Публичные примеры помогают проверить загрузку, рамки, разные масштабы, освещение и перенос фото с экрана телефона. Они не являются собственными фотографиями Софии и не заменяют обучающий набор.',35,H-128,W-70)
text('<b>Как проверять</b><br/>1. Откройте JPG из samples/images или покажите страницу на экране и переснимите телефоном.<br/>2. Запустите обученное приложение на новом снимке. Сравните найденные рамки с различимыми машинами.<br/>3. Запишите пропуски, лишние рамки, марку, цвет и полное время от передачи файла до ответа.<br/>4. Для честной оценки не обучайте сеть на тех же фотографиях, которыми измеряете качество.',35,H-218,W-70)
text('<b>Без готовых ответов</b><br/>Изображения из этой подборки не подключены к модели как база ответов. Марку нужно проверять по видимой эмблеме; имя файла и название модели у автора могут быть неточными. Если эмблема/марка неразличима, не придумывайте эталон. Цвета: white, black, grey, red, blue, green, brown, yellow.',35,190,W-70)
text('<b>Что добавить своими фото:</b> строгий боковой ракурс, задний ракурс, дальние машины, перекрытия, день/вечер; BYD, Chery, Haval, Changan, Geely, Jetour и другие нужные марки. В этой небольшой подборке нет равномерного покрытия марок и цветов.',35,100,W-70,'small')
c.showPage()
for n,r in enumerate(records,2):
    page(n,'EXAMPLE / '+r['id'],r['purpose'])
    image=Image.open(samples/r['image']);maxw,maxh=W-70,H-230;scale=min(maxw/image.width,maxh/image.height);iw,ih=image.width*scale,image.height*scale
    thumb=image.convert('RGB');thumb.thumbnail((1600,1600));thumbfile=temp/(r['id']+'.jpg');thumb.save(thumbfile,quality=90)
    x=(W-iw)/2;y=H-115-ih;c.drawImage(str(thumbfile),x,y,width=iw,height=ih,preserveAspectRatio=True)
    source=escape(r['source'],{'"':'&quot;'});license=escape(r['license_url'],{'"':'&quot;'})
    text(f"Фото: <b>{escape(r['author'])}</b> · {escape(r['date'])} · <a href='{license}' color='#087f8c'>{escape(r['license'])}</a><br/><a href='{source}' color='#087f8c'>Страница источника и оригинальный файл</a> · {escape(r['title'])}<br/>Снимок не редактировался; в PDF пропорционально уменьшен. Права на фото — по указанной лицензии.",35,102,W-70,'small')
    text('Мой результат:  машин ____  пропусков ____  лишних рамок ____  марок верно ____  цветов верно ____  время ____ с',35,46,W-70,'small');c.showPage()
page(8,'REHEARSAL / RECORD','Лист проверки на своих фотографиях')
text('Для каждого нового кадра отдельно проверяйте полноту детекции и правильность марки/цвета. При пересъёмке экрана учитывайте блики, муар, угол и время передачи файла. Число машин само по себе не является метрикой точности.',35,H-125,W-70)
headers=['Фото / ракурс','Эталонных\nмашин','Пропуски','Лишние\nрамки','Верные\nмарки','Верные\nцвета','Время, с']
widths=[230,80,80,85,80,80,135];x=35;y=H-210
c.setFillColorRGB(.08,.18,.27);c.rect(x,y,W-70,34,fill=1,stroke=0);c.setFillColorRGB(1,1,1);c.setFont('ArialBold',9)
for title,width in zip(headers,widths):
    for i,line in enumerate(title.split('\n')):c.drawString(x+7,y+21-i*11,line)
    x+=width
c.setStrokeColorRGB(.78,.82,.87)
for i in range(8):
    rowy=y-i*32;c.line(35,rowy,W-35,rowy)
    if i<7:c.setFillColorRGB(.2,.27,.36);c.setFont('Arial',10);c.drawString(42,rowy-21,f'{i+1:02d}')
text('Авторы и лицензии каждой фотографии сохранены в samples/sources.json. Ссылки в подписях кликабельны. Разметка для обучения создаётся отдельно во вкладке «Разметка»; эта подборка не импортируется в неё автоматически.',35,77,W-70,'small')
c.showPage();c.save()
import shutil
shutil.copy2(output,samples/'Tashkent_Photo_Examples.pdf')
print(output)
