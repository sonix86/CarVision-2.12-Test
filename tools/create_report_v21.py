"""Six-page evidence-based report draft; missing repository/live test explicit."""
import json,sys,collections
from pathlib import Path
from xml.sax.saxutils import escape
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
ROOT=Path(__file__).resolve().parents[1]
def read(f):return json.loads((ROOT/f).read_text(encoding='utf-8-sig'))
v=read('models/release_v21_validation.json');body=read('models/body_expert_evaluation.json');training=read('models/body_expert_training.json');colour=read('models/colour_training.json');paint=read('models/masked_colour_training.json');owner=read('models/owner_adaptation_training.json')
output=ROOT/'output/pdf';output.mkdir(parents=True,exist_ok=True);tmp=ROOT/'outputs/report-v21';tmp.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(11,5));a=np.array(v['confusion']);ax.imshow(a,cmap='Blues');ax.set_xticks(range(len(v['columns'])),v['columns'],rotation=65,ha='right',fontsize=7);ax.set_yticks(range(len(v['brands'])),v['brands'],fontsize=8)
for y in range(a.shape[0]):
 for x in range(a.shape[1]):
  if a[y,x]:ax.text(x,y,str(a[y,x]),ha='center',va='center',fontsize=7,color='white' if a[y,x]>a.max()/2 else 'black')
ax.set_xlabel('Predicted make (unknown is an explicit abstention)');ax.set_ylabel('Reference make');fig.tight_layout();fig.savefig(tmp/'confusion.png',dpi=140);plt.close(fig)
pdfmetrics.registerFont(TTFont('CV','C:/Windows/Fonts/arial.ttf'));pdfmetrics.registerFont(TTFont('CVB','C:/Windows/Fonts/arialbd.ttf'))
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='TextCV',fontName='CV',fontSize=9,leading=12,spaceAfter=7));styles.add(ParagraphStyle(name='TitleCV',fontName='CVB',fontSize=21,leading=25,textColor=colors.HexColor('#14243a'),spaceAfter=14));styles.add(ParagraphStyle(name='SmallCV',fontName='CV',fontSize=7.5,leading=10,spaceAfter=6))
story=[]
def p(s,style='TextCV'):story.append(Paragraph(escape(str(s)).replace('\n','<br/>'),styles[style]))
def title(s):p(s,'TitleCV')
def page(s):story.append(PageBreak());title(s)
def table(rows,widths=None):
 t=Table([[Paragraph(escape(str(c)),styles['SmallCV']) for c in row] for row in rows],colWidths=widths,repeatRows=1)
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcefed')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#b5c5cc')),('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]));story.append(t);story.append(Spacer(1,9))
title('CarVision 2.1\nCar Brand Classification in Tashkent')
p('CSAI-3210 | Midterm | Fall2026 | Report updated22October2026')
p('1. Team, repository and contributions','TitleCV')
p('Participant: Sofia Rakhmankulova (София Рахманкулова).')
p('GitHub repository: NOT PROVIDED YET. This report is a review draft, not a submission-ready PDF. Insert the real repository URL here before submission; no repository has been published by the assistant.')
p('Contribution record: the participant supplied10 phone photos and product requirements. OpenAI Codex assisted with implementation, debugging, public-data collection, local training, packaging, tests and this report. Public-photo creators retain their authorship. No student coding history, label provenance or test results have been fabricated.')
p('Task output: a list of English (brand, colour) pairs, one per visible classifiable car. Concrete model names and top-three alternatives are optional additions. The runtime is offline and requires no API keys, internet service or payment.')
p('Readiness: a portable Windows prototype is implemented. Remaining academic gaps include a sufficiently diverse own-photo training basis, an independent phone-camera rehearsal, the repository URL/tag and confirmation of architectural uniqueness among teams. These gaps cannot be certified by a successful EXE launch.')
p('AI disclosure: substantially AI-assisted work. The participant must understand and explain the code, data, model decisions and limitations; no AI assistance is used during the live test.')
page('2. Data and labels')
p('Body/model data:779 train,161 validation,416 test photos;32 model families. Sources:528 reviewed regional Commons/Flickr crops,634 Stanford Cars crops with original model annotations,194 Chinese/EV crops. Source URLs, authors, licences and hashes are recorded in data and samples manifests. Public sources currently dominate the released weights; this does not close the assignment requirement for own Tashkent data.','SmallCV')
table([['Make','Body-expert training crops']]+sorted(v['training_brand_counts'].items()),[300,210])
table([['Colour','Train','Validation','Test']]+[[c,colour['counts']['train'][c],colour['counts']['validation'][c],colour['counts']['test'][c]] for c in colour['classes']],[210,100,100,100])
p('Colour counts above describe the separate VCoR colour branch, not the body dataset. Silver maps to grey; beige/gold to brown; orange to yellow. Cars in the two datasets must not be summed as unique vehicles.','SmallCV')
p('Own photos:10 originals from one outing,9 manually identified crops (Chevrolet8/Kia1; white7/black2),8 exact-model labels. Ambiguous cars remain unlabelled. All outing frames stay in TRAIN; no neighbouring view of the same car is claimed as independent validation. Their12-epoch experiment was rejected. Public source/duplicate groups retain their historical splits; complete physical-vehicle identity is unavailable for Stanford/VCoR.','SmallCV')
from PIL import Image as PILImage
sample_paths=[]
for index,box in [(1,(247,535,899,969)),(2,(269,509,620,711))]:
 path=tmp/f'own-{index}.jpg';PILImage.open(ROOT/f'data/own_oct21/images/photo_{index}_2026-10-21_13-00-40.jpg').crop(box).save(path);sample_paths.append(path)
story.append(Table([[Image(str(path),width=150,height=90,kind='proportional') for path in sample_paths]],colWidths=[255,255]))
p('Own TRAIN examples: Chevrolet Cobalt and Tracker. Same outing; not an independent test.','SmallCV')
page('3. Architecture and decision rules')
from reportlab.graphics.shapes import Drawing,Rect,String,Line,Polygon
d=Drawing(510,52)
for i,label in enumerate(['Phone photo','Visible masks','Own classifiers','Conflict check','Brand + colour']):
 x=i*103;d.add(Rect(x,12,96,32,fillColor=colors.HexColor('#dcefed'),strokeColor=colors.HexColor('#087f8c')));d.add(String(x+48,24,label,textAnchor='middle',fontName='CV',fontSize=9))
 if i<4:d.add(Line(x+96,28,x+103,28));d.add(Polygon([x+103,28,x+99,31,x+99,25],fillColor=colors.black))
story.append(d)
table([['Stage','Operation'],['Phone JPEG -> laptop','Cable or messaging transfer; no external service during inference'],['Visible instances','General COCO Mask R-CNN; exclusive visible masks; conservative fragment union'],['Make and body/model','Own MobileNetV3-small avg/max-pooled heads and own references; ResNet18 body expert with aspect-preserving224padding'],['Colour','Own colour classifier plus visible-paint features; reflections/lighting can still cause errors'],['Decision consistency','Conflicting make/model outputs -> unknown make; leading model remains a question-mark hypothesis'],['Output','English brand colour for the assignment; JSON/TXT/annotated JPEG and optional model alternatives']],[130,380])
p('ImageNet and COCO initialization are general-purpose only. No downloaded car-brand, car-model or car-colour classifier is used. ResNet18 layers3/4 and its32-class head were trained locally; the older project classifier and train-only reference features are retained as a validation-selected mixture.')
ck=torch.load(ROOT/'models/vehicle_model.pt',weights_only=True);state=ck['body_expert']['state'];parameters=sum(t.numel() for k,t in state.items() if not any(s in k for s in ['running_mean','running_var','num_batches_tracked']))
p(f"Body expert parameter count: {parameters:,}. Total loaded neural parameters: {read('models/release_v21_parameters.json')['loaded_torch_parameters']:,}, including segmentation, make/model and colour networks. Non-neural forest/reference data are excluded. Component counts are recorded in release_v21_parameters.json.")
p('Emblem experiment:5 manually cropped Chevrolet/Kia badges were compared using local contrast templates. Validation acceptance failed; it is disabled. Version2.1 fixes a disabled-module execution bug and a missing-brand crash. No badge-based reliability claim is made.')
p('A high model score and a high make score come from different distributions. They are not directly comparable percentages of truth. Version2.1 discloses disagreement instead of silently displaying a conflicting confirmed make. Colour does not hard-exclude a model.')
page('4. Training and reproducibility')
best=next(r for r in training['history'] if r['epoch']==training['best_epoch'])
table([['Item','Recorded configuration'],['Main expert','Generic ImageNet ResNet18 initialization; own32-family head'],['Loss','Cross entropy; label smoothing0.04'],['Optimiser','AdamW, weight decay0.02; cosine decay'],['Learning rates','layer3:0.000035; layer4:0.00008; head:0.0007'],['Batch / hardware','24 / local CPU,3PyTorch threads'],['Schedule','First experiment stopped after9epochs to fix crop alignment; corrected12epochs; selected epoch'+str(training['best_epoch'])],['Mixture','Expert weight '+str(best['weight'])+'; temperature '+str(best['temperature'])],['Corrected training time',str(round(training['seconds']/60,1))+' minutes; measured script time'],['Augmentation','Mirrors, colour jitter, greyscale, blur, visible masks, upper-body crops and40-96pixel images']],[135,375])
p('Selection used existing grouped validation only. Guards limited loss of full-crop top1/top3 and fragment top3 while improving Chevrolet macro accuracy. Calibration also uses validation. Existing test images were evaluated after selection; they have been used in prior project iterations and are not a newly blinded examination.')
p('Own-photo experiment: frozen project-trained features and12 FC-training epochs, public rehearsal plus8 owner model crops. No candidate passed the validation guards. These rejected weights were not deployed; neither quantity nor quality of own training data is sufficient to claim the academic data requirement is complete.')
p('Training interface: labels include image, box, brand, colour, source and group. Whole sessions/duplicates stay together. Version2.1 writes models/own_training_candidate; it no longer overwrites the production models immediately. Checkpoint, evaluation and detector training were exercised offline with a synthetic engineering fixture; those results are not recognition accuracy.')
page('5. Evaluation and error analysis')
p(f"Current make decision,161 validation body crops: {v['brand_accuracy']:.1%} correct; {v['unknown']} unknown outputs, including {v['conflicts']} head conflicts. Reference classes are the makes available in this body validation set, not every Tashkent brand. No detector AP or full-scene accuracy is implied.")
story.append(Image(str(tmp/'confusion.png'),width=510,height=232))
rows=[['Model top1,416 held-out crops','v1.9','v2.0/2.1 weights']]
for k in ['whole','small64','upper55']:
 r=body['results'][k];rows.append([k,f"{r['baseline']['all']['top1']:.1%}",f"{r['candidate']['all']['top1']:.1%}"])
table(rows,[270,120,120])
p(f"Unchanged colour components: VCoR validation {colour['best_validation_accuracy']:.1%}, test {colour['test_accuracy']:.1%}; masked-paint component validation {paint['validation_accuracy']:.1%}, test {paint['test_accuracy']:.1%}. These component/crop measurements do not establish the colour accuracy of a complete scene.",'SmallCV')
p('Chinese models remain difficult: similar SUV silhouettes, few independent samples, reflections and unfamiliar generations cause wrong candidates. Examples include BYD/Geely/Chery confusion and white paint judged brown under warm lighting. New consistency logic removes contradictory confirmed labels; it does not retrain the model or promise100% accuracy.17 actual development/sample photos ran without exceptions. Phone-camera screen-to-laptop rehearsal has NOT been conducted.','SmallCV')
page('6. How to run; 7. Reflection')
p('Unzip the complete portable archive on Windows10/11 x64 into a writable folder. Keep CarVision.exe, _internal, models, data and samples together. Python is not required for inference. Open a phone photo, click Analyze, then copy the English brand-colour answer. Use mouse region selection only when appropriate; the exam output must cover every classifiable car.')
p('Source setup: Python3.11; install requirements-lock.txt, then run python launch.py gui. Training: python launch.py train --labels data/labels.jsonl --output models/own_training_candidate. Evaluation uses the same labels, saved split and dataset fingerprint. Do not treat examples or development photos as new independent tests.')
p('Engineering checks:78 automated tests including candidate-training GUI routing; end-to-end offline synthetic training/checkpoint/evaluation/detector smoke; actual sample inference regressions. Release checks also cover frozen-source correspondence, a copied EXE without Python in PATH, offline prediction, ROI, exports and archive hashes. See release logs for verification results.')
p('Live-test procedure: photograph the displayed image with the participant phone, transfer to the laptop, run once, write the model output within2minutes. The required20 unseen photos, including at least half Chinese-brand scenes, must remain unseen. A successful local copy test is not a second-physical-computer certification.')
p('Reflection: the weakest areas are data diversity, night/reflection colour, occluded vehicles and open-world Chinese models. Better data and independent vehicle/session validation are necessary. Increasing displayed percentages or memorising a supplied image would not solve these problems. Current own-photo experiments were honestly rejected rather than promoted.')
p('Before submission: insert the real GitHub URL on page1; upload code, labels/data or lawful links, weights or links, environment and this corrected report; create an authentic final midterm tag; verify a clean laptop and phone rehearsal; confirm architecture uniqueness with the instructor; ensure the participant can explain the code. Deadline27October2026,23:59Tashkent; live test28October. The report is intentionally marked DRAFT while these steps remain.')
p('References: original CSAI-3210 assignment; Stanford Cars (Krause et al.,2013); VCoR data provenance in project manifests; Wikimedia/Flickr per-file sources in THIRD_PARTY.md and samples source JSON; torchvision general ImageNet/COCO architectures. Public photographs are not credited to the participant.','SmallCV')
def footer(canvas,doc):
 canvas.setFont('CV',8);canvas.setFillColor(colors.HexColor('#667788'));canvas.drawString(38,22,'CarVision2.1 | Sofia Rakhmankulova | Review draft');canvas.drawRightString(A4[0]-38,22,str(doc.page))
destination=output/'CarVision_Report_v2.1_DRAFT.pdf';doc=SimpleDocTemplate(str(destination),pagesize=A4,leftMargin=38,rightMargin=38,topMargin=34,bottomMargin=36);doc.build(story,onFirstPage=footer,onLaterPages=footer)
from pypdf import PdfReader
assert len(PdfReader(destination).pages)==6,'Report must be exactly6pages'
print(destination)
