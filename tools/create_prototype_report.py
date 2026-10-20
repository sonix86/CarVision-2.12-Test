"""Six-page evidence-based prototype report. Missing academic evidence stays explicit."""
import json,sys
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.graphics.shapes import Drawing,Rect,Line,Polygon,String
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from carvision.core import ROOT,read_image
brand=json.loads((ROOT/'models/evaluation/public_test.json').read_text());colour=json.loads((ROOT/'models/colour_training.json').read_text());training=json.loads((ROOT/'models/public_brand_training.json').read_text());provenance=json.loads((ROOT/'data/public_prototype/provenance.json').read_text());runtime=json.loads((ROOT/'outputs/release-runtime/checks.json').read_text())
pdfmetrics.registerFont(TTFont('CV','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('CVBold','C:/Windows/Fonts/arialbd.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CVBody',fontName='CV',fontSize=9.1,leading=12.5,textColor=colors.HexColor('#27384a'),spaceAfter=7))
styles.add(ParagraphStyle(name='CVTitle',fontName='CVBold',fontSize=24,leading=28,textColor=colors.HexColor('#14243a'),spaceAfter=16))
styles.add(ParagraphStyle(name='CVHeading',fontName='CVBold',fontSize=15,leading=19,textColor=colors.HexColor('#087f8c'),spaceAfter=10))
styles.add(ParagraphStyle(name='CVSmall',fontName='CV',fontSize=7.5,leading=10,textColor=colors.HexColor('#63748a'),spaceAfter=5))
story=[]
def p(text,style='CVBody'):story.append(Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style]))
def heading(text):p(text,'CVHeading')
def page(title):story.append(PageBreak());p(title,'CVTitle')
def table(rows,widths=None):
    items=[[Paragraph(escape(str(v)),styles['CVSmall']) for v in row] for row in rows]
    t=Table(items,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1f1f0')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#d5dee8')),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),2)]));story.append(t);story.append(Spacer(1,8))
def figure(path,width,height):
    im=read_image(path);scale=min(width/im.width,height/im.height);story.append(Image(str(path),width=im.width*scale,height=im.height*scale,hAlign='LEFT'))
def footer(canvas,document):
    canvas.setStrokeColor(colors.HexColor('#d5dee8'));canvas.line(36,35,A4[0]-36,35);canvas.setFont('CV',7);canvas.setFillColor(colors.HexColor('#63748a'));canvas.drawString(36,23,'CarVision / Sofia Rakhmatulaeva / prototype draft / 20.10.2026');canvas.drawRightString(A4[0]-36,23,str(document.page))

p('Tashkent\nCarVision','CVTitle');p('CSAI-3210 Midterm / обученный локальный прототип','CVHeading')
p('Владелец: София Рахматулаева.\nGitHub: полный URL пока не предоставлен владельцем.\nСтатус: ЧЕРНОВИК ПРОТОТИПА, НЕ ОКОНЧАТЕЛЬНАЯ СДАЧА.')
p('Задача: по новой фотографии автоматически найти автомобили и получить список (марка, цвет). Ввод правильной марки в режиме анализа не нужен. Точная модель автомобиля не требуется. Программа работает локально на Windows x64; платные API, ключи и интернет при распознавании не нужны.')
table([['Реальное измерение','Результат'],['Марка, 13 известных классов',f"{brand['brand_top1_accuracy']:.1%}, {brand['images']} эталонных вырезок Stanford Cars"],['Цвет, 8 классов',f"{colour['test_accuracy']:.1%}, {colour['test']} изображений VCoR"],['Основная машина на целых фото','26/26 найдены, правильная принятая марка 17/26; один режим fast'],['Собственные учебные фото Ташкента','0. Публичные данные их не заменяют.']], [190,330])
p('Авторство и вклад: Codex выполнил реализацию, подготовку публичных данных, обучение, программные проверки и упаковку. София указана как владелец проекта. Личный вклад в сбор/проверку данных и индивидуальная рефлексия участника пока не предоставлены; они не выдумываются.')
p('Ограничение интерпретации: показатели марки и цвета получены на разных публичных наборах и не составляют общий процент правильных ответов на улицах Ташкента. Неизвестные марки, отражения, фон, вечер и сильные перекрытия остаются источниками ошибок.')
figure(ROOT/'samples/images/04.jpg',370,170)
credit=next(r for r in json.loads((ROOT/'samples/sources.json').read_text(encoding='utf-8')) if r['id']=='04')
p('Публичный пример входа, не собственные учебные данные. Фото: '+credit['author']+'; '+credit['license']+'. '+credit['source'],'CVSmall')

page('01 / Данные и разметка')
p('Марка: Stanford Cars (Krause et al., 2013), публичные фотографии и исходные человеческие рамки. Изображения зеркала сопоставлены с оригиналами по точному Git blob SHA1; проверены марка и официальный split, конфликты и точные SHA256-дубликаты удалены. Из official train выделены 3102 обучающие и 683 dev-validation машины; official test содержит 4026 машин. Идентичности повторных автомобилей сверх исходного split не доступны.')
table([['Марка','Official train*','Official test']]+[[b,provenance['counts']['train'].get(b,0),provenance['counts']['test'].get(b,0)] for b in brand['brands']],[240,140,140])
p('* Official train затем разбит на gradient train и dev-validation, seed 3210; максимум 600 изображений на марку в исходной подготовке. Китайские марки, Kia и Lada в текущую обученную голову не входят. Разделение списка ручных подсказок GUI и реально обученных классов принципиально.','CVSmall')
table([['Цвет','Train','Val','Test']]+[[c,colour['counts']['train'].get(c,0),colour['counts']['validation'].get(c,0),colour['counts']['test'].get(c,0)] for c in colour['classes']],[220,100,100,100])
p('Цвет: VCoR, исходные 15 цветов сведены к 8: silver -> grey; beige/gold/tan -> brown; orange -> yellow; pink/purple исключены. Исходные train/val/test сохранены, удалены точные SHA256 и одинаковые dHash-дубликаты между частями. Идентичности машин не доступны. Публичные исходники не включены в переносимый пакет; ссылки, метки, хеши и численности сохранены.')

page('02 / Архитектура и обучение')
diagram=Drawing(520,112)
for x,y,w,h,label in [(0,42,70,30,'New photo'),(84,42,100,30,'COCO detector'),(198,42,66,30,'Car crops'),(288,76,115,27,'Brand CNN / 13'),(288,10,115,27,'Features + colour MLP'),(435,42,82,30,'Brand + colour')]:
    diagram.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=colors.HexColor('#e1f1f0'),strokeColor=colors.HexColor('#93bdbd')));diagram.add(String(x+w/2,y+h/2-3,label,fontName='CV',fontSize=7,textAnchor='middle',fillColor=colors.HexColor('#14243a')))
def arrow(x1,y1,x2,y2):
    diagram.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor('#087f8c'),strokeWidth=1));diagram.add(Polygon([x2,y2,x2-4,y2+2,x2-4,y2-2],fillColor=colors.HexColor('#087f8c'),strokeColor=None))
arrow(70,57,84,57);arrow(184,57,198,57)
for y in [89,23]:
    diagram.add(Line(264,57,276,57,strokeColor=colors.HexColor('#087f8c')));diagram.add(Line(276,57,276,y,strokeColor=colors.HexColor('#087f8c')));arrow(276,y,288,y);diagram.add(Line(403,y,419,y,strokeColor=colors.HexColor('#087f8c')));diagram.add(Line(419,y,419,57,strokeColor=colors.HexColor('#087f8c')))
arrow(419,57,435,57);story.append(diagram);story.append(Spacer(1,8))
table([['Этап','Реализация'],['Поиск машин','Общий Faster R-CNN ResNet50 FPN v2 COCO. Car/truck для легковых, пикапов и фургонов; bus не включается. Размер 640/1024.'],['Несколько машин','Полный кадр; overlapping tiles 768 px в balanced; отражение полного кадра в detail. Восстановление координат, отбрасывание внутренних обрезанных рамок, NMS 0.45.'],['Марка на каждой вырезке','MobileNetV3 Small с ImageNet-основой; average + maximum pooling -> 1152 -> 256 Hardswish/dropout 0.3 -> собственная голова 13 марок. Вход 160.'],['Цвет на каждой вырезке','Отдельные общие ImageNet-признаки, 261 оригинальный пространственный RGB/HSV признак; стандартизация по train -> MLP 128 -> 8. Вход 128.'],['Выход','Список марок/цветов, рамки, уверенность и время; марка ниже 0.5 -> unknown. TXT / JSON / JPG.']], [150,370])
p('Обоснование: общий детектор обеспечивает пригодный CPU-прототип без необходимости учить поиск с нуля на нескольких тысячах одиночных машин. Average/max pooling объединяет распределённые и локальные признаки. Цветовая голова отделена от марки: Stanford Cars не имеет надёжных исходных меток цвета. Общие ImageNet/COCO веса разрешены владельцем и заданием; готовый автомобильный brand/colour classifier не использован. Уникальность среди других команд программой не подтверждается.')
p(f"Марка: {len(training['history'])} эпох, AdamW lr 0.0003, weight decay 0.01, cosine LR; inverse-sqrt weighted CE, smoothing 0.03; batch32, flip/random crop/brightness. Лучший checkpoint: epoch {training['best_epoch']}, dev accuracy {training['best_validation_accuracy']:.1%}.")
p(f"Цвет: обучается собственная MLP при замороженной общей основе, AdamW lr 0.001, weight decay 0.01, weighted CE smoothing 0.03; batch64. {len(colour['history'])} эпох, best epoch {colour['best_epoch']}, dev accuracy {colour['best_validation_accuracy']:.1%}. Время MLP {colour['seconds']:.1f} с не включает получение данных и извлечение общих признаков.")
p('Аппаратная платформа: Intel Xeon E3-1230 v3, 16 GB RAM, CPU PyTorch 2.6.0; Python 3.12.14. Готовый пакет содержит runtime и DLL. Полная история и размеры параметров находятся в исходниках/model card; исходный неиспользуемый восьмицветный head brand-сети не выдаётся за обученный.')
card=json.loads((ROOT/'models/model_card.json').read_text());detector=json.loads((ROOT/'models/generic_detector.json').read_text())
p(f"Параметры: общий детектор {detector['parameters']:,}; brand-сеть всего {card['brand_parameters']:,} (включая неиспользуемый старый colour head); отдельная colour MLP {colour['head_parameters']:,}, её общий замороженный backbone {colour['general_backbone_parameters']:,}.",'CVSmall')

page('03 / Марки: результаты')
p(f"Official test, только эталонные вырезки: {brand['brand_top1_accuracy']:.1%} top-1, n={brand['images']}. Test не использован для градиентов или выбора brand checkpoint. На пороге 0.5 принято {next(r for r in brand['thresholds'] if r['threshold']==.5)['coverage']:.1%} вырезок; точность среди принятых {next(r for r in brand['thresholds'] if r['threshold']==.5)['accuracy_when_accepted']:.1%}. Это закрытый набор 13 марок, а не проверка открытого мира.")
figure(ROOT/'models/evaluation/public_brand_confusion.png',515,390)
p('Полный конвейер отдельно проверен на 26 стратифицированных public-test фотографиях: основная размеченная машина найдена во всех, правильная принятая марка - в 17. Другие машины на этих фотографиях могут быть не размечены, поэтому precision всех объектов из этого опыта не вычисляется. Измерение wall time этого запуска включало длинное прерывание сессии; оно не используется как оценка скорости.')
p('Известные ошибки на городских примерах: локальные варианты Chevrolet/Daewoo отличаются от машин публичного набора. Неизвестная Lada может быть ошибочно отнесена к Volkswagen; unknown по порогу не гарантирует отказ для всех неизвестных марок. Китайские марки не обучены. Для расширения покрытия нужны новые проверенные метки и собственные съёмочные сессии.')

page('04 / Цвет: результаты и ошибки')
p(f"VCoR official test после очистки: {colour['test_accuracy']:.1%}, n={colour['test']}. Это исходные метки набора; все восемь требуемых цветов представлены. На validation сравнивались варианты признаков; сохранён исходный комбинированный вариант. Готовые веса автомобильного colour classifier не использованы.")
figure(ROOT/'models/evaluation/colour_confusion.png',420,330)
p('Основные путаницы: grey с brown/white, metallic beige/gold с grey, тёмный blue с black, red/orange. На городском серебристом автомобиле возможен brown, а отражение голубого неба на чёрном кузове может дать blue. Корректная рамка помогает, но не устраняет освещение и фон. Это наблюдаемые ошибки; они не маскируются принудительными ответами для файлов-примеров.')
p('Маленькая визуальная подборка Stanford, размеченная Codex, использовалась как диагностическая обратная связь разработки. Она не является независимым человеческим финальным эталоном и не подменяет VCoR benchmark или собственный phone-test. Псевдометки HSV в исходном labels.jsonl Stanford не использованы для обучения этой головы.')
p('Следующие технические шаги: собственные дневные/вечерние фото Ташкента, разные ракурсы каждой машины, ручная проверка цвета/эмблемы, группировка по машине и сессии, расширение китайских марок. Раздел «Учебная разметка» сохраняет эти данные; при анализе новой фотографии марку и цвет вводить не требуется.')

page('05 / Запуск и готовность к сдаче')
p('Перенос: полностью распаковать TashkentCarVision-Portable.zip на Windows 10/11 x64 в папку с правом записи; запустить CarVision.exe. Отдельные Python, API-ключи и интернет не нужны. Не отделять EXE от _internal/models/data/samples. Открыть фото -> «Анализировать» -> прочитать/скопировать список или экспортировать TXT/JSON/JPG.')
p(f"Программные проверки: 22 теста; source inference на шести городских JPG с заблокированным socket.connect, повтор после переименования файла; реальные trained checkpoints. Быстрый режим на данных шести фотографиях: {min(r['seconds'] for r in runtime['photos']):.2f}-{max(r['seconds'] for r in runtime['photos']):.2f} с на данном CPU, без времени передачи с телефона. EXE также проверяется в другой папке и без Python в PATH; полный инженерный протокол - VALIDATION.md.")
table([['Перед окончательной сдачей','Статус'],['Собственные данные Ташкента','Не предоставлены; публичный прототип их не заменяет.'],['GitHub URL на первой странице, code/data links/weights','URL ожидается. Локальные исходники, веса, метки и источники готовы.'],['Уникальность family + multi-car strategy','Нужна сверка с преподавателем/другими командами.'],['Реальный телефон -> передача -> ответ <=120 с','Полный цикл не проведён; нужно измерить на целевом ноутбуке.'],['20 невиденных фото и очный live test','Выполняются участником на экзамене; не подтверждаются этим отчётом.'],['Вклады, личная reflection, финальный tag midterm','Нужны фактические сведения участника и окончательная версия.']], [230,290])
p('AI disclosure: Codex помог с кодом, дизайном, подготовкой публичных данных, обучением, экспериментами и документацией. Не создавал фиктивных студентских коммитов и не выдаёт AI-разметку/публичные снимки за собственные фотографии участника. Участник должен понимать весь код и пройти очный тест без AI/посторонней помощи.')
p('Источники: Stanford Cars / Krause et al. 2013, huggingface.co/datasets/tanganke/stanford_cars и github.com/jhpohovey/StanfordCars-Dataset; VCoR / Panetta et al. 2021, doi.org/10.3390/ai2040041 и Kaggle landrykezebou/vcor-vehicle-color-recognition-dataset; torchvision 0.21 model documentation. Авторы/CC-лицензии шести городских примеров: samples/sources.json.','CVSmall')
p('Срок задания: финальная версия с midterm до 27.10.2026 23:59 Ташкент; live test 28.10.2026, NH311. Этот честный черновик должен быть обновлён после получения недостающих данных и GitHub.','CVSmall')
out=ROOT/'output/pdf/CarVision_Prototype_Report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
document=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=36,rightMargin=36,topMargin=35,bottomMargin=48,title='CarVision prototype report - draft',author='Sofia Rakhmatulaeva; AI assistance disclosed')
document.build(story,onFirstPage=footer,onLaterPages=footer)
if document.page!=6:raise ValueError('Expected exactly six pages; got '+str(document.page))
print('Created',out,'pages',document.page)
