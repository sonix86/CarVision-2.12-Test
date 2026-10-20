"""Produce a six-page report from measured results; never invent missing evidence."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from .core import ROOT, COLOURS

def report(args):
    required=[ROOT/'models/training.json',ROOT/'models/evaluation/metrics.json',ROOT/'data/labels.jsonl',Path(args.analysis),ROOT/'models/detector_training.json']
    for f in required:
        if not f.exists():raise FileNotFoundError(f'Report requires actual evidence: {f}')
    if not args.repo.startswith('https://github.com/'):raise ValueError('Provide a full real GitHub repository URL')
    train=json.loads(required[0].read_text(encoding='utf-8')); metrics=json.loads(required[1].read_text(encoding='utf-8'))
    analysis=json.loads(Path(args.analysis).read_text(encoding='utf-8'))
    detector=json.loads(required[4].read_text(encoding='utf-8'))
    for key in ['brand_choice','collection_labelling','architecture_rationale','errors','chinese_brands','phone_tests','reflection']:
        if not analysis.get(key):raise ValueError('Analysis JSON must include actual '+key)
    styles=getSampleStyleSheet(); styles['BodyText'].fontSize=9; styles['BodyText'].leading=12
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import os
    fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    if (fonts/'arial.ttf').exists():
        pdfmetrics.registerFont(TTFont('CVArial',str(fonts/'arial.ttf')))
        pdfmetrics.registerFont(TTFont('CVArialBold',str(fonts/'arialbd.ttf')))
        for s in styles.byName.values():s.fontName='CVArial'
        styles['Heading1'].fontName='CVArialBold';styles['Heading2'].fontName='CVArialBold'
    story=[]
    def para(text,style='BodyText'):story.append(Paragraph(escape(str(text)).replace('\n','<br/>'),styles[style]));story.append(Spacer(1,6))
    def heading(text):para(text,'Heading1')
    heading('Car Brand Classification in Tashkent')
    para('CSAI-3210 | Midterm | Fall 2026')
    para('Team: '+args.team);para('Repository: '+args.repo);para('Contributions: '+args.contributions)
    para('AI-use disclosure: OpenAI Codex assisted in requirements analysis, code implementation, packaging, documentation and automated tests. Team members must review, adapt and understand the code. Data collection, label verification, reported results and individual contributions must be attributed to their actual authors.')
    para('1. Data','Heading2');para(analysis['brand_choice']);para(analysis['collection_labelling']);para('Sources and counts: '+json.dumps(train['data']['sources']))
    para('Groups of repeated vehicles/sessions and exact duplicate images are kept together. 80/20 group split, seed '+str(train['seed'])+'. Validation is used for checkpoint selection. No claim of an independent final test.')
    story.append(PageBreak());heading('2. Data distribution and samples')
    rows=[['Brand','All','Train','Validation']]+[[b,train['data']['brands'][b],train['train']['brands'].get(b,0),train['validation']['brands'].get(b,0)] for b in sorted(train['data']['brands'])]
    def table(rows):
        t=Table(rows,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17354d')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.3,colors.lightgrey),('FONTSIZE',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]));story.append(t);story.append(Spacer(1,8))
    table(rows)
    table([['Colour','All','Train','Validation']]+[[c,train['data']['colours'].get(c,0),train['train']['colours'].get(c,0),train['validation']['colours'].get(c,0)] for c in COLOURS])
    labels=[json.loads(s) for s in required[2].read_text(encoding='utf-8').splitlines() if s.strip()]
    # Small labelled crops, with original provenance retained in labels.jsonl.
    from .core import read_image
    from io import BytesIO
    samples=[]
    for r in labels[:3]:
        im=read_image(ROOT/'data'/r['image']).crop(r['box']);im.thumbnail((130,85));buf=BytesIO();im.save(buf,format='PNG');buf.seek(0)
        samples.append([Image(buf,width=im.width,height=im.height),Paragraph(escape(r['brand']+' '+r['colour']),styles['BodyText'])])
    if samples:story.append(Table([[Table([[s[0]],[s[1]]]) for s in samples]],colWidths=[160]*len(samples)))
    story.append(PageBreak());heading('3. Architecture and training')
    para('Design rationale: '+analysis['architecture_rationale'])
    # An actual diagram made of cells/arrows, not an unrendered Mermaid block.
    for text in ['Phone photo (cable / offline transfer)','↓','Team-trained SSDLite320 MobileNetV3 Large → full view + overlapping tiles → NMS','↓','Per-car crop → MobileNetV3 Small → average + maximum pooling','↓','1152 → 256 Hardswish / dropout 0.3','↓','Brand head + eight-colour head → (brand, colour) list']:
        para(text)
    para('Both networks are trained from random initialization on team data, without COCO or ImageNet weights. Low-confidence brand outputs unknown (threshold 0.65). Detector threshold 0.5; NMS IoU 0.45. Balanced mode uses 768-pixel overlapping tiles; detail mode also uses a mirrored full view. Classifier parameters: '+str(train['parameters'])+'; detector parameters: '+str(detector['parameters'])+'.')
    para('Training: '+train['loss']+'; '+train['optimizer']);para('Learning rate '+str(train['lr'])+'; epochs '+str(train['epochs'])+'; batch '+str(train['batch'])+'; hardware '+train['hardware']+'; '+str(round(train['seconds'],1))+' seconds.');para('Augmentation: '+train['augmentation'])
    para('Detector: AdamW, LR '+str(train['lr'])+', batch 2, '+str(detector['epochs'])+' epochs, '+str(round(detector['seconds'],1))+' seconds; SSD classification and box-regression losses, horizontal flips. Detector held-out metrics: '+json.dumps(detector['history'][-1]))
    story.append(PageBreak());heading('4. Evaluation')
    para(f"Held-out car crops: {metrics['validation_cars']}. Brand accuracy {metrics['brand_accuracy']:.3f}; colour accuracy {metrics['colour_accuracy']:.3f}.")
    para(metrics['scope'])
    matrix=ROOT/'models/evaluation/brand_confusion.png'
    if matrix.exists():story.append(Image(str(matrix),width=390,height=350,kind='proportional'))
    para(analysis['errors']);para('Chinese brands: '+analysis['chinese_brands'])
    story.append(PageBreak());heading('5. Phone-camera rehearsal and test day')
    para(analysis['phone_tests'])
    para('Unzip the entire portable folder to a writable local directory. Start CarVision.exe. Transfer the phone photograph by cable (or another method allowed by the instructor). Open the photo, run recognition once, copy the output from the list or TXT file to the answer sheet. The machine performs inference locally; no API, AI assistant or external help is used during the live test. Keep the complete folder and models together. Measure end-to-end timing including transfer, aiming below 120 seconds.')
    para('Reproduce training: python launch.py train --labels data/labels.jsonl --epochs 25. Evaluate: python launch.py evaluate --labels data/labels.jsonl --output models/evaluation. Exact runtime versions are in requirements-lock.txt. Follow README.md for clean installation and building the EXE.')
    story.append(PageBreak());heading('6. Reflection, provenance and integrity')
    para(analysis['reflection'])
    para('The architecture must be checked against other teams by the students/instructor; software alone cannot certify uniqueness. Only original team code and team-trained brand/colour weights are submitted. No fabricated student commits or contributions are generated.')
    para('Network implementations use torchvision; no pretrained weights are loaded. Documentation: https://docs.pytorch.org/vision/stable/models.html . Library provenance and public example-photo licenses are recorded in THIRD_PARTY.md and samples/sources.json. Public examples are not automatically imported as training data.')
    para('Submit one PDF, no more than six pages, with the repository URL on its first page. The repository must contain code, labels, data or a data link, trained weights or a download link, and a meaningful real commit history. Tag the verified final commit midterm before 27 October 2026, 23:59 Tashkent time. Live test: 28 October 2026, NH311.')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_name(output.stem+'.review.tmp.pdf')
    document=SimpleDocTemplate(str(temporary),pagesize=A4,rightMargin=36,leftMargin=36,topMargin=30,bottomMargin=30)
    document.build(story)
    if document.page>6:raise ValueError(f'Report has {document.page} pages; limit is 6. Shorten analysis/tables and regenerate before submission.')
    temporary.replace(output)
    print('Report generated. Render and visually inspect every page; verify <= 6 pages before submission: '+str(output))
