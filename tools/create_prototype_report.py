"""Six-page v1.5 prototype report; missing academic evidence remains explicit."""
import sys,json,collections
from pathlib import Path
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image
from reportlab.graphics.shapes import Drawing,Rect,Line,String
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
card=read('models/model_card.json');fusion=read('models/mask_fusion.json');model=read('models/instance_vehicle_calibration.json');adapt=read('models/instance_vehicle_training.json');paint=read('models/paint_forest_experiment.json');mask_training=read('models/masked_colour_training.json');metrics=read('models/evaluation/instance_metrics.json');runtime=read('outputs/release-runtime-v15/checks.json');segmenter=read('models/segmenter.json')
pdfmetrics.registerFont(TTFont('CV','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('CVBold','C:/Windows/Fonts/arialbd.ttf'))
styles=getSampleStyleSheet()
for name,size,leading,font,colour in [('CV',9,12,'CV','#243746'),('Small',7.2,9.4,'CV','#62758a'),('CVTitle',23,27,'CVBold','#14243a'),('Heading',13,16,'CVBold','#087f8c')]:
 styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,textColor=colors.HexColor(colour),spaceAfter=7))
story=[]
def p(text,style='CV'):
 text=str(text).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-')
 story.append(Paragraph(escape(text).replace('\n','<br/>'),styles[style]))
def page(title):story.append(PageBreak());p(title,'CVTitle')
def table(rows,widths):
 t=Table([[Paragraph(escape(str(v)),styles['Small']) for v in r] for r in rows],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1f1f0')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d6e0e8')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]));story.extend([t,Spacer(1,6)])
def report_asset(path):
    path=Path(path)
    if path.is_relative_to(ROOT/'samples/images'):
        folder=ROOT/'tmp/pdfs/v15/assets';folder.mkdir(parents=True,exist_ok=True);destination=folder/(path.stem+'.jpg')
        if not destination.exists() or destination.stat().st_mtime<path.stat().st_mtime:
            image=read_image(path);image.thumbnail((1100,1100));image.save(destination,quality=88,optimize=True)
        return destination
    return path
def figure(path,width,height):
 path=report_asset(path);im=read_image(path);s=min(width/im.width,height/im.height);story.append(Image(str(path),width=im.width*s,height=im.height*s,hAlign='LEFT'))
def footer(canvas,document):
 canvas.setStrokeColor(colors.HexColor('#d6e0e8'));canvas.line(36,34,A4[0]-36,34);canvas.setFont('CV',7);canvas.drawString(36,22,'CarVision 1.5 / Sofia Rakhmankulova / prototype draft / 21.10.2026');canvas.drawRightString(A4[0]-36,22,str(document.page))
p('Tashkent\nCarVision 1.5','CVTitle');p('CSAI-3210 Midterm / локальный обученный прототип','Heading')
p('Владелец: София Рахманкулова.\nGitHub: полный URL ещё не предоставлен владельцем.\nСтатус: ЧЕРНОВИК ПРОТОТИПА, НЕ ОКОНЧАТЕЛЬНАЯ СДАЧА.')
p('По фотографии программа находит видимые контуры машин и выдаёт пары (марка, цвет). Дополнение владельца: до трёх вероятных конкретных моделей. Правильные ответы вводить не нужно. Windows x64, локальное выполнение, без платных сервисов, ключей и интернет-запросов при распознавании.')
table([['Измерение','Результат'],['Марка / видимые маски Commons',f"{metrics['brand_test_accuracy']:.1%},80 вырезок,7 марок представлены"],['Модель / чистые вырезки',f"top-1 {model['metrics']['test']['clean']['top1']:.1%}; top-3 {model['metrics']['test']['clean']['top3']:.1%}; n80"],['Модель / искусственное перекрытие',f"top-1 {model['metrics']['test']['partial']['top1']:.1%}; top-3 {model['metrics']['test']['partial']['top3']:.1%}; n80"],['Цвет / маски VCoR',f"{paint['test_accuracy']:.1%};240 сбалансированных машин"],['Собственные учебные фото','0. Публичные данные не заменяют это требование.']], [205,315])
p('Это разные публичные протоколы на вырезках; их проценты не являются общей точностью улиц Ташкента. Искусственное перекрытие не равно реальной окклюзии. Шесть городских примеров служат диагностикой разработки и не считаются независимым финальным тестом.')
p('Вклад: Codex выполнил реализацию, подготовку публичных данных, обучение, инженерные проверки и упаковку. София указана как владелец; личный сбор/проверка данных и индивидуальная рефлексия пока не предоставлены и не выдумываются.')
figure(ROOT/'samples/images/06.jpg',370,150)
p('Публичный пример входа, не собственные данные. Фото и полная атрибуция: samples/sources.json; индивидуальные CC-лицензии сохранены.','Small')

page('01 / Данные и разметка')
public=[json.loads(s) for s in (ROOT/'data/public_prototype/labels.jsonl').read_text(encoding='utf-8').splitlines()]
regional=[json.loads(s) for s in (ROOT/'data/regional/labels.jsonl').read_text(encoding='utf-8').splitlines()]
split=read('models/public_split.json')
sets={k:set(split.get(k,[r['image'] for r in public if r.get('official_split')=='test'] if k=='test' else [])) for k in ['train','validation','test']}
p('Stanford Cars: исходные человеческие рамки/марки, сверка оригинальных аннотаций по Git blob SHA1, очистка точных дубликатов. Commons: явные категории/названия, общие автоматические рамки, фильтры неоднозначности/размера и near-dHash. Группировка по автору/дате; истинные vehicle IDs недоступны. Маски не добавляют новых фотографий.','Small')
rows=[['Марка','Stanford train / val / test','Commons train / val / test']]
for brand in card['supported_brands']:
 a=[sum(r['brand']==brand and r['image'] in sets[s] for r in public) for s in ['train','validation','test']]
 b=[sum(r['brand']==brand and r['split']==s for r in regional) for s in ['train','validation','test']]
 rows.append([brand,' / '.join(map(str,a)),' / '.join(map(str,b))])
table(rows,[120,200,200])
manifest=read('data/public_prototype/colour_dataset_manifest.json')['rows']
rows=[['Цвет','Полный VCoR train / val / test','Масочный поднабор']]
for colour in card['colour_classes']:
 rows.append([colour,' / '.join(str(sum(r['colour']==colour and r['split']==s for r in manifest)) for s in ['train','validation','test']),'80 / 20 / 30'])
table(rows,[120,200,200])
p('VCoR: исходные цветовые метки,15 классов сведены к8; silver→grey, beige/gold/tan→brown, orange→yellow. Pink/purple исключены. Исходный split сохранён, exact SHA256 и одинаковые dHash очищены. Псевдометки HSV Stanford не обучают цвет. Масочный train640/val160/test240 выбран детерминированно внутри прежних частей.','Small')
pics=[]
for name in ['01','03','06']:
 im=read_image(ROOT/'samples/images'/f'{name}.jpg');scale=min(155/im.width,78/im.height);pics.append(Image(str(report_asset(ROOT/'samples/images'/f'{name}.jpg')),width=im.width*scale,height=im.height*scale))
story.append(Table([pics],colWidths=[170,170,170]));p('Публичные примеры входа; авторы, URL и лицензии в samples/sources.json. Сами примеры не обучают сеть.','Small')

page('02 / Архитектура и обучение')
d=Drawing(520,95)
for x,y,w,h,text in [(0,32,78,30,'Phone / region'),(92,32,105,30,'COCO visible masks'),(211,32,77,30,'Isolated cars'),(305,57,100,28,'Own brand/model'),(305,7,100,28,'Own paint colour'),(431,32,86,30,'Brand + colour')]:
 d.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=colors.HexColor('#e1f1f0'),strokeColor=colors.HexColor('#93bdbd')));d.add(String(x+w/2,y+h/2-2,text,fontName='CV',fontSize=7,textAnchor='middle'))
for coords in [(78,47,92,47),(197,47,211,47),(288,47,305,71),(288,47,305,21),(405,71,431,47),(405,21,431,47)]:d.add(Line(*coords,strokeColor=colors.HexColor('#087f8c')))
story.append(d)
table([['Этап','Реализация'],['Контур',f"Общий COCO Mask R-CNN ResNet50 FPN v2, {segmenter['parameters']:,} параметров. Car/truck, фильтр длинных truck-подобных ложных областей. Масочная дедупликация; пиксель закреплён за одной машиной."],['Несколько машин / зона','Весь кадр либо зона. Для одной машины сохраняется полный контекст перекрытий; маска ограничивается выбранной зоной. При промахе: увеличенная зона, затем приблизительный GrabCut с предупреждением. Скрытые части не реконструируются.'],['Марка / модель','Собственные MobileNetV3 Small avg/max1152→256→18 марок и отдельная16-модельная сеть. Марка использует изолированный вид; модель смешивает полный/изолированный виды0.75/0.25. Температура/порог по dev; три вероятных модели даже при неустановленной марке.'],['Цвет','Собственные прежние VCoR-головы + собственный лес64 деревьев,572 HSV/Lab признака видимых пикселей, min_leaf3. sklearn только при обучении; EXE использует экспортированные массивы деревьев.'],['Выход','Контуры и номера, варианты в панели, отдельная легенда JPG, все маски/проценты в JSON. Основной TXT и кнопка задания выдают brand colour; подробности - details.txt.']], [120,400])
p('Выбор дизайна: общий сегментатор отделяет кузова при перекрытии, а собственные головы сохраняют ответственность за марку/цвет. Частоты цвета считаются по маске; новые варианты не заменяют обучение неизвестных классов. COCO/ImageNet разрешены; готовый автомобильный brand/model/colour classifier не использован.')
table([['Обучение','Фактические настройки'],['Марка','База40 эпох AdamW0.0003; региональная адаптация18 эпох feature0.00006/head0.0003, wd0.015, batch32, CE smoothing0.05, cosine. Crop55-100%, flip/jitter/erasing. Исходные истории в models/*.json.'],['Модель, этап1','40 эпох AdamW feature0.00012/head0.001, wd0.02, batch32. Split374/82/80. Затем новый этап ниже.'],['Модель, этап2',f"24 эпохи, лучший epoch{adapt['best_epoch']}; AdamW feature0.00006/head0.0003, wd0.015, CE smoothing0.05, batch32, cosine.60% изолированных видов,50% искусственных перекрытий; crop65-100%, flip/jitter. Время {adapt['seconds']:.0f}с с подготовкой масок."],['Новый цвет',f"640 train,160 validation; собственная MLP160 эпох проверена, выбран лес64 деревьев по validation. Время подготовки/MLP {mask_training['seconds']:.0f}с; лес {paint['seconds']:.1f}с. Смесь и пороги выбраны без test/городских примеров."]], [120,400])
p('CPU: Xeon E3-1230 v3,16 GB RAM; PyTorch2.6.0+cpu, Python3.12.14. Головы марки/модели около1.23M параметров каждая; старый восьмицветный head этих сетей не выдаётся за обученный. Все точные эпохи, функции потерь, параметры и эксперименты сохранены в models/*.json.','Small')

page('03 / Марки, модели и ошибки')
p(f"Марка по видимой маске: {metrics['brand_test_accuracy']:.1%},80 Commons-вырезок. В тесте представлены7 из18 марок. Предыдущий Stanford benchmark80.4% на4026 полных эталонных вырезках измерял другой протокол.")
figure(ROOT/'models/evaluation/instance_brand_confusion.png',515,325)
p('Китайские марки: BYD, Chery и Haval представлены малым числом публичных примеров. '+ '; '.join(f"{b}: {v['correct']}/{v['test_count']} в этом тесте" for b,v in metrics['chinese'].items())+'. Выводы по четырём машинам ненадёжны. Changan, Geely, Jetour и многие другие пока не обучены; похожие SUV и небольшие эмблемы остаются источником ошибок. Собственные фотографии новых китайских марок - главный следующий шаг.','CV')
table([['Конкретная модель / n80','Top-1','Правильная среди3'],['Чистая масочная вырезка',f"{model['metrics']['test']['clean']['top1']:.1%}",f"{model['metrics']['test']['clean']['top3']:.1%}"],['Искусственное перекрытие27%',f"{model['metrics']['test']['partial']['top1']:.1%}",f"{model['metrics']['test']['partial']['top3']:.1%}"],['До адаптации / перекрытие',f"{adapt['baseline']['test']['partial']:.1%}",'отдельные базовые метрики в mask_fusion.json']], [260,100,160])
p(f"Checkpoint выбран только по validation с ограничением на чистые вырезки. Температура {model['temperature']}, порог модели {model['threshold']}. Это закрытые16 вариантов. Damas имеет лишь3 train, Chevrolet Nexia4 train/1 test, Cobalt12 train/2 test без dev. Их уличная надёжность не установлена.")
p('Номера на фото и контуры - локализация, не гарантия верного названия. Неизвестная модель может получить уверенное известное имя. Перекрытые/маленькие машины, поколения одного имени и несовпадение эмблемы с предположением требуют осторожности.','Small')

page('04 / Цвет и ошибки')
p(f"Новый масочный протокол: validation {paint['validation_accuracy']:.1%},160 машин; test {paint['test_accuracy']:.1%},240 машин, по30 каждого цвета. Охват уверенных {paint['test_coverage']:.1%}, правильных среди них {paint['test_accepted_accuracy']:.1%}. Старые93.6% на1314 полных VCoR-изображениях не являются тем же тестом.")
figure(ROOT/'models/evaluation/instance_colour_confusion.png',450,330)
p('Маски исключают соседний кузов из цветовой области.1035 из1040 подготовленных границ дал общий COCO,5 - GrabCut. Лес и нейронные головы обучены в проекте по исходным VCoR-цветам. Ни готовый colour classifier, ни псевдометки HSV Stanford не подменяют эталон.')
p('Средняя точность на этом поднаборе не выросла относительно нейронного baseline изолированных вырезок:91.7% в обоих случаях. Практическое изменение - устранение смешивания машин в одном прямоугольнике. На присланной сцене соседняя с зелёной Lada машина больше не получает зелёный от её кузова. Эта сцена диагностическая, не независимый тест.')
p('Остались путаницы white/grey/brown, beige/gold с grey, тёмного blue с black; тени и отражения могут менять видимый оттенок. Сомнительный цвет помечается «≈», второй вариант виден в деталях. Качество отдельной маски тоже может быть низким; полная оценка по новым телефонным фото ещё нужна.')

page('05 / Запуск и готовность к сдаче')
p('Распаковать TashkentCarVision-v1.5-Portable.zip целиком на Windows10/11 x64 в папку с правом записи. Запустить CarVision.exe. Переносить всю папку _internal/models/data/samples вместе с EXE; Python/API/интернет для распознавания не нужны.')
p('На тесте: снять фото экрана своим телефоном → передать файлом на ноутбук → «Открыть фото» → «Анализировать фото» → скопировать пары «Марки / цвета для задания». Одна попытка до просмотра эталона. Модели и альтернативы - дополнительная информация; правильную марку вручную не выбирать.')
p(f"43 инженерных теста; реальные веса, запрет socket.connect, переименование фото без изменения ответов, зоны и маски. Шесть диагностических JPG: {min(r['seconds'] for r in runtime['photos']):.2f}-{max(r['seconds'] for r in runtime['photos']):.2f}с на данном CPU без передачи с телефона. Проверка EXE/переносимой копии и SHA256 - VALIDATION.md.")
table([['Перед окончательной сдачей','Статус'],['Собственные данные Ташкента','Пока0 учебных фото. Публичный прототип это требование не заменяет.'],['GitHub URL на странице1, веса/данные по ссылкам','URL ожидается; локальные исходники, метрики и веса готовы.'],['Уникальность family + multi-car strategy','Нужна фактическая сверка с преподавателем/командами.'],['Телефон → передача → ответ ≤120с','Полный цикл не измерен; указано только локальное время.'],['20 невиденных фото, другой физический ноутбук','Очный тест/второе устройство ещё не подтверждены.'],['Личный вклад/reflection, финальный tag midterm','Нужны фактические сведения участника и окончательный репозиторий.']], [225,295])
p('AI disclosure: Codex помог с кодом, дизайном, подготовкой публичных данных, обучением, экспериментами, тестами и документацией. София - владелец проекта. Фиктивные студенческие коммиты/вклады не создавались. Участник должен понимать код и пройти очный тест без AI/посторонней помощи.')
p('План на ещё две недели, предложенный AI: собственные дневные/вечерние фото Ташкента, проверенные человеком марки/цвета, больше китайских брендов и реальных перекрытий, группировка по машине/сессии, полный phone-test. Личная reflection участника ещё ожидается.','Small')
p('Источники: Stanford Cars/Krause2013 (huggingface.co/datasets/tanganke/stanford_cars и github.com/jhpohovey/StanfordCars-Dataset); VCoR/Panetta2021, doi.org/10.3390/ai2040041, Kaggle landrykezebou/vcor-vehicle-color-recognition-dataset; torchvision0.21 COCO Mask R-CNN. Индивидуальные Commons URL/авторы/CC-лицензии - data/regional/source_manifest_snapshot.json и samples/sources.json.','Small')
p('Срок:27.10.2026 23:59 Ташкент, tag midterm; live test28.10.2026, NH311. Данный черновик не выдаётся за полностью закрытое задание.','Small')
out=ROOT/'output/pdf/CarVision_Prototype_Report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=36,rightMargin=36,topMargin=35,bottomMargin=47,title='CarVision1.5 prototype report - draft',author='Sofia Rakhmankulova; Codex assistance disclosed')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
if doc.page!=6:raise ValueError('Expected exactly6 pages; got '+str(doc.page))
print('Created',out,'pages',doc.page)
