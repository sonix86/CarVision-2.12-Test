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
brand=json.loads((ROOT/'models/evaluation/public_test.json').read_text(encoding='utf-8'));colour=json.loads((ROOT/'models/colour_training.json').read_text(encoding='utf-8'));training=json.loads((ROOT/'models/public_brand_training.json').read_text(encoding='utf-8'));provenance=json.loads((ROOT/'data/public_prototype/provenance.json').read_text(encoding='utf-8'));runtime=json.loads((ROOT/'outputs/release-runtime/checks.json').read_text(encoding='utf-8'))
regional=json.loads((ROOT/'models/regional_brand_training.json').read_text(encoding='utf-8'));regional_data=json.loads((ROOT/'data/regional/provenance.json').read_text(encoding='utf-8'));card=json.loads((ROOT/'models/model_card.json').read_text(encoding='utf-8'))
pdfmetrics.registerFont(TTFont('CV','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('CVBold','C:/Windows/Fonts/arialbd.ttf'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='CVBody',fontName='CV',fontSize=9.1,leading=12.5,textColor=colors.HexColor('#27384a'),spaceAfter=7))
styles.add(ParagraphStyle(name='CVTitle',fontName='CVBold',fontSize=24,leading=28,textColor=colors.HexColor('#14243a'),spaceAfter=16))
styles.add(ParagraphStyle(name='CVHeading',fontName='CVBold',fontSize=15,leading=19,textColor=colors.HexColor('#087f8c'),spaceAfter=10))
styles.add(ParagraphStyle(name='CVSmall',fontName='CV',fontSize=7.5,leading=10,textColor=colors.HexColor('#63748a'),spaceAfter=5))
vehicle=json.loads((ROOT/'models/vehicle_model_training.json').read_text(encoding='utf-8'));calibration=json.loads((ROOT/'models/colour_calibration.json').read_text(encoding='utf-8'))
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
    canvas.setStrokeColor(colors.HexColor('#d5dee8'));canvas.line(36,35,A4[0]-36,35);canvas.setFont('CV',7);canvas.setFillColor(colors.HexColor('#63748a'));canvas.drawString(36,23,'CarVision / Sofia Rakhmatulaeva / prototype draft / 21.10.2026');canvas.drawRightString(A4[0]-36,23,str(document.page))

p('Tashkent\nCarVision','CVTitle');p('CSAI-3210 Midterm / обученный локальный прототип','CVHeading')
p('Владелец: София Рахматулаева.\nGitHub: полный URL пока не предоставлен владельцем.\nСтатус: ЧЕРНОВИК ПРОТОТИПА, НЕ ОКОНЧАТЕЛЬНАЯ СДАЧА.')
p('Задача: по новой фотографии автоматически найти автомобили и получить список (марка, цвет). Ввод правильной марки в режиме анализа не нужен. Версия 1.4 дополнительно определяет конкретную модель для изученных вариантов; при недостаточной уверенности она остаётся неопределённой. Программа работает локально на Windows x64; платные API, ключи и интернет при распознавании не нужны.')
table([['Реальное измерение','Результат'],['Марка / Stanford Cars',f"{brand['brand_top1_accuracy']:.1%}, {brand['images']} эталонных вырезок; 13 марок представлены в этом тесте"],['Марка / местные варианты',f"{regional['regional_test_accuracy']:.1%}, {regional['test_regional']} отложенных публичных вырезок Commons"],['Модель автомобиля',f"{vehicle['test_accuracy']:.1%} на {vehicle['test_count']} вырезках; {len(vehicle['labels'])} вариантов, публичные данные"],['Цвет, 8 классов',f"{calibration['test_accuracy']:.1%}, {calibration['test_count']} изображений VCoR"],['Собственные учебные фото Ташкента','0. Публичные данные их не заменяют.']], [190,330])
p('Авторство и вклад: Codex выполнил реализацию, подготовку публичных данных, обучение, программные проверки и упаковку. София указана как владелец проекта. Личный вклад в сбор/проверку данных и индивидуальная рефлексия участника пока не предоставлены; они не выдумываются.')
p('Ограничение интерпретации: показатели марки и цвета получены на разных публичных наборах и не составляют общий процент правильных ответов на улицах Ташкента. Неизвестные марки, отражения, фон, вечер и сильные перекрытия остаются источниками ошибок.')
figure(ROOT/'samples/images/04.jpg',370,145)
credit=next(r for r in json.loads((ROOT/'samples/sources.json').read_text(encoding='utf-8')) if r['id']=='04')
p('Публичный пример входа, не собственные учебные данные. Фото: '+credit['author']+'; '+credit['license']+'. '+credit['source'],'CVSmall')

page('01 / Данные и разметка')
p('Марка: Stanford Cars (Krause et al., 2013), публичные фотографии и исходные человеческие рамки. Изображения зеркала сопоставлены с оригиналами по точному Git blob SHA1; проверены марка и официальный split, конфликты и точные SHA256-дубликаты удалены. Из official train выделены 3102 обучающие и 683 dev-validation машины; official test содержит 4026 машин. Идентичности повторных автомобилей сверх исходного split не доступны.')
table([['Марка','Official train*','Official test']]+[[b,provenance['counts']['train'].get(b,0),provenance['counts']['test'].get(b,0)] for b in brand['brands'] if provenance['counts']['test'].get(b,0)],[240,140,140])
p(f"* Базовый train/dev: seed3210. Commons: train {regional['train_regional']}, validation {regional['validation_regional']}, test {regional['test_regional']}. Явные категории/названия, общий детектор рамок; исключены неоднозначные сцены, near-dHash дубликаты и похожие на шесть примеров снимки. Автор/дата группируются; идентичности машин не гарантированы. Региональные train/val/test: "+'; '.join(f"{b} {v.get('train',0)}/{v.get('validation',0)}/{v.get('test',0)}" for b,v in regional_data['counts'].items())+'.','CVSmall')
table([['Цвет','Train','Val','Test']]+[[c,colour['counts']['train'].get(c,0),colour['counts']['validation'].get(c,0),colour['counts']['test'].get(c,0)] for c in colour['classes']],[220,100,100,100])
p('Цвет: VCoR, исходные 15 цветов сведены к 8: silver -> grey; beige/gold/tan -> brown; orange -> yellow; pink/purple исключены. Исходные train/val/test сохранены, удалены точные SHA256 и одинаковые dHash-дубликаты между частями. Идентичности машин не доступны. Публичные исходники не включены в переносимый пакет; ссылки, метки, хеши и численности сохранены.')

page('02 / Архитектура и обучение')
diagram=Drawing(520,112)
for x,y,w,h,label in [(0,42,70,30,'Photo / region'),(84,42,100,30,'COCO detector'),(198,42,66,30,'Car crops'),(288,76,115,27,f"Own brand + model CNNs"),(288,10,115,27,'Own colour decisions'),(435,42,82,30,'Vehicle + colour')]:
    diagram.add(Rect(x,y,w,h,rx=4,ry=4,fillColor=colors.HexColor('#e1f1f0'),strokeColor=colors.HexColor('#93bdbd')));diagram.add(String(x+w/2,y+h/2-3,label,fontName='CV',fontSize=7,textAnchor='middle',fillColor=colors.HexColor('#14243a')))
def arrow(x1,y1,x2,y2):
    diagram.add(Line(x1,y1,x2,y2,strokeColor=colors.HexColor('#087f8c'),strokeWidth=1));diagram.add(Polygon([x2,y2,x2-4,y2+2,x2-4,y2-2],fillColor=colors.HexColor('#087f8c'),strokeColor=None))
arrow(70,57,84,57);arrow(184,57,198,57)
for y in [89,23]:
    diagram.add(Line(264,57,276,57,strokeColor=colors.HexColor('#087f8c')));diagram.add(Line(276,57,276,y,strokeColor=colors.HexColor('#087f8c')));arrow(276,y,288,y);diagram.add(Line(403,y,419,y,strokeColor=colors.HexColor('#087f8c')));diagram.add(Line(419,y,419,57,strokeColor=colors.HexColor('#087f8c')))
arrow(419,57,435,57);story.append(diagram);story.append(Spacer(1,8))
table([['Этап','Реализация'],['Поиск машин','Общий Faster R-CNN ResNet50 FPN v2 COCO; car/truck, bus исключён. Размер 800/1333. По умолчанию весь кадр.'],['Область / несколько машин','Можно искать внутри области либо напрямую распознать одну обведённую машину. Координаты возвращаются к исходному фото. В balanced/detail дополнительные фрагменты, отбрасывание вложенных фрагментов и NMS0.45.'],['Марка','MobileNetV3 Small, average/max pooling ->1152->256-> собственные марки. Вход160; исходный/зеркальный вид, усреднение вероятностей, температура по development.'],['Цвет',colour['method']],['Выход','Номера на рамках, марка, модель и цвет в списке; марка ниже0.5 -> unknown. TXT/JSON/JPG, отдельная легенда без перекрытия машин.']], [150,370])
p('Обоснование: общий детектор обеспечивает пригодный CPU-прототип без необходимости учить поиск с нуля на нескольких тысячах одиночных машин. Average/max pooling объединяет распределённые и локальные признаки. Цветовая голова отделена от марки: Stanford Cars не имеет надёжных исходных меток цвета. Общие ImageNet/COCO веса разрешены владельцем и заданием; готовый автомобильный brand/colour classifier не использован. Уникальность среди других команд программой не подтверждается.')
p(f"Марка: {len(training['history'])} эпох, AdamW lr 0.0003, weight decay 0.01, cosine LR; inverse-sqrt weighted CE, smoothing 0.03; batch32, flip/random crop/brightness. Лучший checkpoint: epoch {training['best_epoch']}, dev accuracy {training['best_validation_accuracy']:.1%}.")
p(f"Региональная адаптация: {len(regional['history'])} эпох, AdamW backbone0.00006/head0.0003, weight decay0.015, cosine; weighted CE smoothing0.05; batch32. Crop55-100%, flip, colour jitter и erasing3-25%. Checkpoint выбирается по двум dev-наборам при Stanford dev>=80%. Температура {regional['temperature']:.2f}.")
p(f"Цвет: {colour['optimizer']}; batch {colour['batch']}, {colour['loss']}. Выбор по VCoR validation и 23 отдельным development-цветам; tests не выбирают веса. Метод/истории/параметры: models/colour_training.json и colour_candidate_comparison.json.")
p('Аппаратная платформа: Intel Xeon E3-1230 v3, 16 GB RAM, CPU PyTorch 2.6.0; Python 3.12.14. Готовый пакет содержит runtime и DLL. Полная история и размеры параметров находятся в исходниках/model card; исходный неиспользуемый восьмицветный head brand-сети не выдаётся за обученный.')
card=json.loads((ROOT/'models/model_card.json').read_text(encoding='utf-8'));detector=json.loads((ROOT/'models/generic_detector.json').read_text(encoding='utf-8'))
p(f"Параметры: детектор {detector['parameters']:,}; brand-сеть {card['brand_parameters']:,} (включая неиспользуемый старый colour head); colour heads {colour['head_parameters']:,}, их feature backbones {colour['general_backbone_parameters']:,}.",'CVSmall')

p(f"Модель (дополнение 1.4): своя сеть MobileNetV3 Small, {len(vehicle['labels'])} классов, инициализация собственной сетью марки, 40 эпох AdamW, групповой split 374/82/80; выбран epoch {vehicle['best_epoch']} по dev. Test top-1 {vehicle['test_accuracy']:.1%}; при пороге {vehicle['threshold']:.2f} охват {vehicle['test_coverage']:.1%}, точность принятых {vehicle['test_accepted_accuracy']:.1%}. Для Damas/Nexia мало примеров; оценки не доказывают уличную точность.",'CVSmall')

page('03 / Марки: результаты')
p(f"Official test, только эталонные вырезки: {brand['brand_top1_accuracy']:.1%} top-1, n={brand['images']}. Test не использован для градиентов или выбора brand checkpoint. На пороге 0.5 принято {next(r for r in brand['thresholds'] if r['threshold']==.5)['coverage']:.1%} вырезок; точность среди принятых {next(r for r in brand['thresholds'] if r['threshold']==.5)['accuracy_when_accepted']:.1%}. Это закрытый набор 13 марок, а не проверка открытого мира.")
figure(ROOT/'models/evaluation/public_brand_confusion.png',515,390)
p(f"Новый региональный holdout: {regional['regional_test_accuracy']:.1%} против {regional['baseline_regional_test_accuracy']:.1%} у v1.2 на тех же вырезках. Половина автомобиля (левая/правая50%): {regional['regional_half_vehicle_test_accuracy']:.1%} против {regional['baseline_regional_half_vehicle_test_accuracy']:.1%}. Это искусственная обрезка отложенных рамок, не реальная окклюзия на улице.")
p('Шесть городских примеров, включая присланные сцены, служат регрессионной диагностикой разработки; их цвета/марки не подставляются по имени файла. Они не являются независимым экзаменационным тестом. Новые классы представлены меньшим числом снимков; сильные перекрытия, малые эмблемы и неизвестные марки остаются риском.')

page('04 / Цвет: результаты и ошибки')
p(f"VCoR official test после очистки: {colour['test_accuracy']:.1%}, n={colour['test']}. Все восемь требуемых цветов представлены. Отдельная development-диагностика23 сложных вырезок: {colour['cross_domain_development_accuracy']:.1%}; она использована при выборе, поэтому не считается независимым финальным тестом. Готовые vehicle-colour weights не использованы.")
figure(ROOT/'models/evaluation/colour_confusion.png',420,330)
p('Основные путаницы: grey с brown/white, metallic beige/gold с grey, тёмный blue с black, red/orange. На городском серебристом автомобиле возможен brown, а отражение голубого неба на чёрном кузове может дать blue. Корректная рамка помогает, но не устраняет освещение и фон. Это наблюдаемые ошибки; они не маскируются принудительными ответами для файлов-примеров.')
p('Маленькая визуальная подборка Stanford, размеченная Codex, использовалась как диагностическая обратная связь разработки. Она не является независимым человеческим финальным эталоном и не подменяет VCoR benchmark или собственный phone-test. Псевдометки HSV в исходном labels.jsonl Stanford не использованы для обучения этой головы.')
p('Следующие технические шаги: собственные дневные/вечерние фото Ташкента, разные ракурсы каждой машины, ручная проверка цвета/эмблемы, группировка по машине и сессии, расширение китайских марок. Раздел «Учебная разметка» сохраняет эти данные; при анализе новой фотографии марку и цвет вводить не требуется.')

page('05 / Запуск и готовность к сдаче')
p('Перенос: полностью распаковать TashkentCarVision-v1.4-Portable.zip на Windows10/11 x64 в папку с правом записи; запустить CarVision.exe. Python, API-ключи и интернет не нужны. Сохранять EXE вместе с _internal/models/data/samples. Выбрать «Одна машина в области» -> обвести автомобиль -> «Анализировать»; марку/цвет вводить не нужно.')
p(f"35 программных тестов; source inference на шести городских JPG с заблокированным socket.connect, переименование файла и оба режима области. Fast: {min(r['seconds'] for r in runtime['photos']):.2f}-{max(r['seconds'] for r in runtime['photos']):.2f} с на данном CPU без передачи с телефона. EXE проверяется в другой папке без Python в PATH; VALIDATION.md.")
table([['Перед окончательной сдачей','Статус'],['Собственные данные Ташкента','Не предоставлены; публичный прототип их не заменяет.'],['GitHub URL на первой странице, code/data links/weights','URL ожидается. Локальные исходники, веса, метки и источники готовы.'],['Уникальность family + multi-car strategy','Нужна сверка с преподавателем/другими командами.'],['Реальный телефон -> передача -> ответ <=120 с','Полный цикл не проведён; нужно измерить на целевом ноутбуке.'],['20 невиденных фото и очный live test','Выполняются участником на экзамене; не подтверждаются этим отчётом.'],['Вклады, личная reflection, финальный tag midterm','Нужны фактические сведения участника и окончательная версия.']], [230,290])
p('AI disclosure: Codex помог с кодом, дизайном, подготовкой публичных данных, обучением, экспериментами и документацией. Не создавал фиктивных студентских коммитов и не выдаёт AI-разметку/публичные снимки за собственные фотографии участника. Участник должен понимать весь код и пройти очный тест без AI/посторонней помощи.')
p('Источники: Stanford Cars / Krause et al. 2013, huggingface.co/datasets/tanganke/stanford_cars и github.com/jhpohovey/StanfordCars-Dataset; VCoR / Panetta et al. 2021, doi.org/10.3390/ai2040041 и Kaggle landrykezebou/vcor-vehicle-color-recognition-dataset; torchvision0.21 model documentation. Дополнительные фото: commons.wikimedia.org; индивидуальные URL, авторы, CC-лицензии и изменения - data/regional/source_manifest_snapshot.json. Авторы шести городских примеров - samples/sources.json.','CVSmall')
p('Срок задания: финальная версия с midterm до 27.10.2026 23:59 Ташкент; live test 28.10.2026, NH311. Этот честный черновик должен быть обновлён после получения недостающих данных и GitHub.','CVSmall')
out=ROOT/'output/pdf/CarVision_Prototype_Report.pdf';out.parent.mkdir(parents=True,exist_ok=True)
document=SimpleDocTemplate(str(out),pagesize=A4,leftMargin=36,rightMargin=36,topMargin=35,bottomMargin=48,title='CarVision prototype report - draft',author='Sofia Rakhmatulaeva; AI assistance disclosed')
document.build(story,onFirstPage=footer,onLaterPages=footer)
if document.page!=6:raise ValueError('Expected exactly six pages; got '+str(document.page))
print('Created',out,'pages',document.page)
