"""Evidence-linked six-page report for the active release, without legacy metrics."""
import csv,json,sys,collections,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape
from tools.audit_ev_baseline_v212 import predict_features
from tools.train_truth_v30 import valid

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repository');parser.add_argument('--output-stem',default='CarVision-2.12-Report');args=parser.parse_args()
    torch.set_num_threads(3)
    out=ROOT/'output/pdf';out.mkdir(parents=True,exist_ok=True)
    tmp=ROOT/'tmp/pdfs/report-v212';tmp.mkdir(parents=True,exist_ok=True)
    app=ROOT/'outputs/CarVision-2.12-Test/CarVision'
    ck=torch.load(app/'models/dino_make.pt',weights_only=True,map_location='cpu')
    rows=json.loads((ROOT/'cache/audited_v212/rows.json').read_text('utf-8'))
    d=torch.load(ROOT/'cache/truth_v31/features.pt',weights_only=True)
    brands=ck['brands'];n=len(rows);vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='validation']);vr=[rows[int(i)] for i in vi]
    geom=torch.cat([valid(vr,k) for k in ('whole','phone')]);fg=torch.cat([valid(vr,k,True) for k in ('whole','phone')])
    with torch.inference_mode():p,a=predict_features(ck,d['x'][torch.cat([vi,vi+n])],geom,fg)
    y=torch.tensor([brands.index(r['brand']) for r in vr]*2);pred=p.argmax(1);m=len(vr)
    stats={}
    for j,k in enumerate(('whole','phone')):
        sl=slice(j*m,(j+1)*m);correct=pred[sl]==y[sl];acc=a[sl]
        stats[k]=dict(n=m,raw_top1=float(correct.float().mean()),correct=int((correct&acc).sum()),wrong=int((~correct&acc).sum()),unknown=int((~acc).sum()))
    cm=np.zeros((len(brands),len(brands)+1),dtype=int)
    for t,q,accepted in zip(y[:m].tolist(),pred[:m].tolist(),a[:m].tolist()):cm[t,q if accepted else len(brands)]+=1
    with (out/'CarVision-2.12-confusion.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['actual / predicted']+brands+['unknown']);w.writerows([[b]+cm[i].tolist() for i,b in enumerate(brands)])
    normalized=cm/np.maximum(1,cm.sum(1,keepdims=True));fig,ax=plt.subplots(figsize=(8,7.1));im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(83));ax.set_xticklabels([str(i+1) for i in range(82)]+['U'],fontsize=4,rotation=90)
    ax.set_yticks(range(82));ax.set_yticklabels(range(1,83),fontsize=4)
    ax.set_xlabel('Predicted brand index (page 2); U = unknown',fontsize=8);ax.set_ylabel('True brand index (page 2)',fontsize=8)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02).ax.tick_params(labelsize=7);fig.tight_layout();fig.savefig(tmp/'confusion.png',dpi=240);plt.close(fig)
    colour=json.loads((ROOT/'models/colour_training.json').read_text('utf-8'))
    own=[json.loads(s) for s in (ROOT/'data/reviewed_owner_v212/labels.jsonl').read_text('utf-8').splitlines() if s.strip()]
    source=collections.Counter(r['image'].split('/')[0] for r in rows)
    counts={b:collections.Counter(r['split'] for r in rows if r['brand']==b) for b in brands}
    with (out/'CarVision-2.12-data-counts.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['brand','train','validation','development_test','all']);w.writerows([[b,counts[b]['train'],counts[b]['validation'],counts[b]['test'],sum(counts[b].values())] for b in brands])
    (out/'CarVision-2.12-report-evidence.json').write_text(json.dumps(dict(validation=stats,split_counts=dict(collections.Counter(r['split'] for r in rows)),sources=dict(source),own_reviewed_objects=len(own),scope='Reused development corpus; no blind phone accuracy'),indent=2),'utf-8')
    # Three real reviewed training crops. No predictions are used as their labels.
    samples=[]
    from PIL import Image as PILImage
    for b in ('BYD','Zeekr','Leapmotor'):
        r=next(r for r in own if r['brand']==b);im=PILImage.open(ROOT/'data/reviewed_owner_v212'/r['image']).convert('RGB').crop(r['box']);im.thumbnail((280,150));path=tmp/(b+'.jpg');im.save(path);samples.append((path,b+' | '+r['colour']))
    for name,file in [('CV','arial.ttf'),('CVB','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyCV',fontName='CV',fontSize=9,leading=12,spaceAfter=6,textColor=colors.HexColor('#203349')))
    styles.add(ParagraphStyle(name='SmallCV',fontName='CV',fontSize=7.4,leading=9.5,spaceAfter=4,textColor=colors.HexColor('#43576c')))
    styles.add(ParagraphStyle(name='HeadingCV',fontName='CVB',fontSize=16,leading=20,spaceAfter=10,textColor=colors.HexColor('#12364c')))
    styles.add(ParagraphStyle(name='SubCV',fontName='CVB',fontSize=10.5,leading=14,spaceAfter=6,textColor=colors.HexColor('#087f8c')))
    story=[];plain=[]
    def para(t,style='BodyCV'):
        story.append(Paragraph(escape(t).replace('\n','<br/>'),styles[style]));plain.append(t+'\n')
    def heading(t):para(t,'HeadingCV')
    def sub(t):para(t,'SubCV')
    def table(data,widths=None,size=8):
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT');t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'CV'),('FONTNAME',(0,0),(-1,0),'CVB'),('FONTSIZE',(0,0),(-1,-1),size),('LEADING',(0,0),(-1,-1),size+2),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173b50')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eef5f7'),colors.white]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#087f8c')),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2),('VALIGN',(0,0),(-1,-1),'MIDDLE')]));story.append(t);story.append(Spacer(1,7));plain.extend([' | '.join(map(str,r))+'\n' for r in data])
    def page():story.append(PageBreak())
    # PAGE 1
    heading('CarVision: offline car recognition in Tashkent')
    para('CSAI-3210 Introduction to Artificial Intelligence | Webster University Tashkent\nMidterm report | Fall 2026 | CarVision 2.12 TEST | '+('28 October 2026' if args.repository else '27 October 2026'),'SmallCV')
    sub('1. Team, repository and contributions')
    para('Student: Sofia Rakhmankulova. The user reports that Sofia performed all student project work; no second contributor has been identified. Reported responsibilities: project coordination, collecting local photographs, running the application and reviewing errors. The student must verify this attribution before submission.')
    para('Repository: '+args.repository+'\nRelease: v2.12-test. This publication is dated 28 October 2026, after the stated 27 October 23:59 deadline; no retroactive timely-submission claim is made.' if args.repository else 'Repository status: no GitHub repository URL has been provided. This copy is a technical draft; the required full repository URL must be added to this first page before submission.')
    sub('2. Data: scope and provenance')
    para('The objective is a list of English (brand, colour) pairs for every visible car in a phone photograph, including Chinese EVs, multiple cars and partial views. Exact model recognition is optional. The active classifier has 82 brand labels and a separate 43-family head. The 325-name catalogue is a requested scope, not 325 trained models; several sparse brands cannot obtain a confirmed prediction under calibration.')
    para('The audited brand feature corpus contains 3,851 car crops: 2,205 training, 704 validation and 942 development-test records. It combines public Stanford Cars and attributed Wikimedia/Flickr vehicle photos with a small local phone subset. Public data dominate, so the requirement to rely primarily on own Tashkent phone data has not yet been satisfied.')
    para('The current reviewed local export contains 44 objects in 43 available photographs, 13 brands and three outing/source groups; all are TRAIN-only. Missing original files are excluded. This export overlaps historical training data and must not be added to 3,851 as an independent new sample count. Six newer reviewed training objects supplied 36 augmented feature views for the scoped correction head.')
    para('Labels use source metadata, visible badges, original-image review and user corrections. An early GAC Aion attribution was corrected to Dongfeng/Aeolus E70 after comparison with independently labelled source images. Predictions are never accepted as ground truth. Some imported public labels remain provisional; the audit does not certify every visual label.')
    sub('Reviewed local training examples')
    cells=[]
    for path,label in samples:
        from PIL import Image as PILImage
        with PILImage.open(path) as im:w,h=im.size
        cells.append(Table([[Image(str(path),width=145,height=h*145/w)],[Paragraph(escape(label),styles['SmallCV'])]],colWidths=[155]))
    story.append(Table([cells],colWidths=[166]*3))
    para('Examples shown above were used for development/training; they are not a blind test. Original provenance remains in data/reviewed_owner_v212/labels.jsonl.','SmallCV')
    page()
    # PAGE 2
    heading('2. Data distribution and split (continued)')
    para('All 82 labels in checkpoint order. T = training; V = validation; D = reused development-test. Counts are cached crop records, not unique physical vehicles. Brand indices also identify the confusion matrix on page 5.','SmallCV')
    entries=[[str(i+1),b,counts[b]['train'],counts[b]['validation'],counts[b]['test']] for i,b in enumerate(brands)]
    table([['#','Brand','T','V','D','#','Brand','T','V','D']]+[entries[i]+entries[i+41] for i in range(41)],[20,128,31,31,31,20,128,31,31,31],7.1)
    para('Total: 3,851 = 2,205 T + 704 V + 942 D. Source-prefix counts: '+', '.join(k+' '+str(v) for k,v in source.items())+'.','SmallCV')
    para('Public subsets retain recorded groups; matching source hashes and exact pixels/masks must remain in one split. The 2.12 structural audit found no group, source-hash or exact-pixel cross-split leakage in these 3,851 records. Near-duplicates and repeated physical vehicles without an identity may remain. Repeated use for model development prevents calling D an independent final test.','SmallCV')
    page()
    # PAGE 3
    heading('3. Architecture')
    para('Multi-car instance masks isolate body evidence from streets and neighbouring cars. General visual backbones supply complementary shape/detail descriptors; small supervised heads are feasible to adapt on CPU. This is the actual 2.12 route, rather than the legacy MobileNet-only design.')
    pipeline=[['Phone photo -> local resize / optional tiles and mirrored view'],['COCO Mask R-CNN ResNet50-FPN-v2 -> visible car/truck masks'],['Mask-aware crop -> frozen DINOv2 ViT-S/14 + frozen CLIP ViT-B/32'],['768-D DINO descriptor + 512-D CLIP -> normalized 1,280-D features'],['Own 82-brand cosine head + grouped reference fallback + calibrated gates'],['Own 43-family head; scoped make correction for Zeekr / Leapmotor / BYD'],['Parallel own colour ensemble + masked paint / illumination evidence'],['Per-instance English (brand, colour) list -> GUI / JSON / TXT / annotated JPG']]
    table([['Phone-to-answer pipeline']]+pipeline,[504],8.7)
    para('Each cosine head learns class vectors, biases and a logit scale. Probability, margin, absolute similarity, crop geometry and grouped validation support determine whether a result is confirmed. The family head operates with the accepted make; its manufacturer fallback is limited to Dongfeng when the primary make abstains. Low confidence yields unknown or a visibly unconfirmed hypothesis, not a guaranteed answer.')
    para('Parameter counts: the make head has 105,043 trainable parameters; the 43-family head has '+str(sum(v.numel() for v in ck['model_head']['head'].values()))+'; the additional scoped make head has '+str(sum(v.numel() for v in ck['make_correction_head']['head'].values()))+'. DINO encoder state contains 22,056,576 elements. The paired CLIP visual state contains '+str(sum(v.numel() for v in torch.load(app/'models/semantic_make.pt',weights_only=True)['encoder_state'].values()))+' elements (state counts, not the whole application). Encoders and detector are frozen in the deployed classifier training.')
    para('Colour is independent of make: own VCoR-trained CNN/statistical heads predict eight classes, then visible-paint and scene-illumination evidence are fused. White, black, grey, red, blue, green, brown and yellow are the output vocabulary. Warm light, reflections and windows still cause errors.')
    sub('Originality and changes in 2.12')
    para('Only general-purpose COCO/DINO/CLIP initialization is used. No third-party ready-made car-brand or car-colour classifier and no external inference API are used. Architectural uniqueness relative to classmates must be checked by the student/instructor.')
    para('Version 2.12 fixes the detector-only route selecting the wrong classifier, default colour loading, the joint-colour API, paired-encoder hash checks, conflicting labels/duplicate splits and unsafe reuse of a training output directory. Recognition weights are unchanged from 2.11. These engineering fixes are not evidence of an accuracy increase.')
    page()
    # PAGE 4
    heading('4. Training and 5. Evaluation')
    para('Training used PyTorch CPU execution with three worker threads in the logged head-fitting scripts. Frozen features were cached. Weighted cross-entropy counters imbalance; label smoothing and public-teacher KL distillation limit forgetting. Views include original/mask-aware crops, synthetic phone degradation, horizontal flips, darker colour casts and family edge cuts. Augmentations stay with their source group.')
    table([['Stage','Recorded configuration'],['82-brand adaptation','AdamW; LR 0.0001 joint; wd 0.02; CE smoothing 0.01;\nlocal weight 12; public KL 0.2; selected epoch 80 / 100'],['43-family adaptation','Only three corrected family vectors adapted; AdamW;\nround 1 LR 0.0001; selected epoch 100; other vectors frozen'],['Scoped correction','Two 80-epoch fits, batch 128; LR 0.00005 / 0.0001;\nwd 0.01; smoothing 0.01; local weights 12 / 16; KL distillation'],['Masked paint head','AdamW 0.001; wd 0.03; batch 64; 160 epochs;\nsmoothing 0.03; selected epoch 116; recorded 532.7 seconds']],[110,394],8)
    para('Validation chooses candidates and calibrates rejection. Both unrestricted new make replacements failed guards; only the calibrated scoped correction was retained. The original encoders were not retrained. The aggregate elapsed time for the complete development history was not recorded; the paint-head time above is one component, not total training time.','SmallCV')
    sub('Brand: complete routed classifier, crop-level results')
    vw=stats['whole'];vp=stats['phone']
    table([['Cohort / view','N','Raw top-1','Correct / N','Wrong','Unknown'],['Validation / whole',m,f"{100*vw['raw_top1']:.1f}%",f"{vw['correct']}/{m}",vw['wrong'],vw['unknown']],['Validation / synthetic phone',m,f"{100*vp['raw_top1']:.1f}%",f"{vp['correct']}/{m}",vp['wrong'],vp['unknown']],['Development-test / whole',942,'83.1%','626/942 (66.5%)',23,293],['Development-test / synthetic phone',942,'76.5%','478/942 (50.7%)',24,440],['Target-make subset / whole',174,'76.4%','90/174 (51.7%)',4,80],['Target-make subset / synthetic phone',174,'63.8%','47/174 (27.0%)',6,121]],[189,32,62,99,48,74],7.4)
    para('Correct/N counts abstentions as failures. Raw top-1 ignores rejection. Confirmed precision on D/whole is 626/649 = 96.5%, which is not 96.5% overall recognition. The 20-make target subset includes mixed-powertrain manufacturers and is not 174 certified pure EVs. All cohorts are reused development evidence; synthetic phone views are not new real phone photographs.','SmallCV')
    sub('Colour data and component validation')
    cs=['white','black','grey','red','blue','green','brown','yellow']
    table([['Colour']+cs,['VCoR train']+[colour['counts']['train'][c] for c in cs],['VCoR validation']+[colour['counts']['validation'][c] for c in cs],['VCoR test']+[colour['counts']['test'][c] for c in cs],['Reviewed local TRAIN']+[sum(r['colour']==c for r in own) for c in cs]],[112]+[49]*8,7.2)
    para('Recorded source-domain colour ensemble: validation 94.24% (1,319 crops), test 93.61% (1,314). Masked-paint component: validation 93.75% (160), test 92.08% (240). These are component results, not current end-to-end Tashkent colour accuracy. No independent current full-pipeline colour validation has been completed.','SmallCV')
    page()
    # PAGE 5
    heading('5. Confusion, phone tests and error analysis')
    para('Current routed make confusion on all 704 validation crops, whole view; row-normalized. All 82 classes are retained, including sparse classes; zero-support rows are blank. Numeric class labels map to page 2; U is abstention. Full integer counts are supplied in the companion CSV. This matrix was regenerated from the active 2.12 checkpoint.','SmallCV')
    story.append(Image(str(tmp/'confusion.png'),width=480,height=426))
    para('Chinese-brand failures are driven by similar sedan/SUV bodies, limited local rear/night views, small or absent badges, crop/background sensitivity and sparse group support. The E70/GAC mix-up also exposed a reference-label mistake. A high similarity bar is not a calibrated probability of correctness; forcing the top class would convert unknowns into confident mistakes.')
    para('Known real-photo diagnostics: Zeekr and Leapmotor make corrections now pass, but the exact Zeekr 007 model is not trained. Voyah and a white Chevrolet Tracker remain wrong/unconfirmed; AIQAR is outside the 82-brand vocabulary. A white BYD Dolphin may be coloured brown. Some parked cars remain too small or occluded. These known cases became training/development evidence and cannot establish generalization.')
    para('Historical local rehearsal: 99 photographs were processed, but only a 41-object reviewed subset had usable make/colour references. Version 2.10 achieved 37/41 confirmed makes and 26/41 colours on that known development subset. This is not a 2.12 blind-test score or accuracy over all 99 photos. No defensible prediction of “8 out of 10 new photos” is available.')
    para('Engineering verification: 150 unit tests passed; final EXE analysis of six known photographs took 14.8-16.3 seconds per process on this PC, excluding transfer. EXE help, crop selection/export, paired weights and the detector-only route were checked with Python removed from PATH. A second physical PC and a full timed exam rehearsal have not yet been tested.','SmallCV')
    page()
    # PAGE 6
    heading('6. How to run and 7. Reflection')
    sub('Exact test-day procedure')
    para('1. Extract the complete CarVision 2.12 Full archive into a new writable Windows 10/11 x64 folder. Keep CarVision.exe, models and _internal together. Do not install old patches over it. No Python installation is needed for the packaged application.\n2. Run CHECK_ON_NEW_PC.cmd once, then disable Wi-Fi and verify a photograph. PASS checks execution, not answer accuracy.\n3. Launch CarVision.exe and confirm the title says 2.12 TEST. Transfer the exam photo by the permitted local method; USB avoids dependence on Telegram/internet.\n4. Choose Open photo, then Analyse photo once. Review every visible-car row. Copy the English brand/colour pairs using the assignment-answer button; do not substitute a speculative alternative as a confirmed result.\n5. Record the answer before seeing the reference. Preserve JSON/TXT/annotated output. Measure the complete transfer-to-answer cycle against the exam time limit.')
    sub('Command-line alternative inside the application folder')
    para('CarVision.exe predict "C:\\photos\\car.jpg" --mode fast --output outputs\\exam\nCHECK_ON_NEW_PC.cmd','SmallCV')
    para('For development only, after installing requirements: python launch.py predict "C:\\photos\\car.jpg" --mode fast --output outputs\\exam. The older EXAM/midterm wrapper is optional and is not the primary GUI path. Personal training must write to a fresh output folder; it does not automatically update the advanced 82-brand head. Training scripts and recorded selection policies document the specialist lineage.')
    sub('Reflection and AI-use statement')
    para('With two more weeks, the priority is independently verified local phone data: distinct vehicles, makes, front/rear/side views, occlusion and evening light. Reserve an untouched vehicle/session-separated test before further fitting. Train balanced body and badge/detail specialists with hard negative pairs, improve visible-paint/illumination handling, and report detection recall, make/colour accuracy, unknown rate and the actual exam score. These are proposed improvements, not implemented results or an 80% guarantee.')
    para('OpenAI Codex assisted with requirements analysis, code implementation, debugging, experiment scripts, packaging, automated verification and preparation of this report. Human-provided photos, feedback and corrections were used, but some labels required independent rechecking. Sofia must review and understand the submitted code, verify the contribution statement and report actual work honestly. No AI assistant or external service is called during inference.')
    para('Publication: '+args.repository+'. Code, release restore instructions, trained weights, available data/crop caches, licences and evidence are supplied. The midterm tag identifies the real publication snapshot, not a claim of submission before the deadline. Original local history is preserved without invented or backdated commits. Student understanding and architectural uniqueness still require instructor verification.' if args.repository else 'Submission status: add the real GitHub URL to page 1; provide code, labels, lawful data/weight links and requirements; verify clean-machine instructions; create the real final midterm tag before the deadline. Genuine contribution history cannot be reconstructed by invented commits. The report does not certify a completed repository submission or uniqueness across teams.','SmallCV')
    sub('References and reproducibility evidence')
    refs=['[1] Assignment: CSAI-3210 Midterm, Fall 2026, sections 4-7; locally extracted in outputs/assignment-source.txt.',
    '[2] Generic detector: pytorch.org/vision/stable/models/mask_rcnn.html; DINOv2: github.com/facebookresearch/dinov2; CLIP visual initialization: huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K.',
    '[3] Stanford Cars: ai.stanford.edu/~jkrause/cars/car_dataset.html. Wikimedia/Flickr author, licence and source records: dataset manifests and THIRD_PARTY.md. VCoR: kaggle.com/datasets/landrykezebou/vcor-vehicle-color-recognition-dataset. Observe each source licence; do not publish private photos without permission.',
    '[4] Measurements: models/phone_v32/regression.json; outputs/ev-specialist-v212/baseline.json; models/colour_training.json; models/masked_colour_training.json. Current validation matrix regenerated by tools/report_v212.py.',
    '[5] Integrity: outputs/pipeline-audit-v212/audit.json; outputs/tests-audit-v212-final.log; outputs/exe-v212-verification/verification.json. Active make SHA256: c1b01c33d55e163815734ba8c37810da97ac02bba0f9d042519635815d9fac14.']
    for t in refs:para(t,'SmallCV')
    def chrome(c,doc):
        c.setStrokeColor(colors.HexColor('#087f8c'));c.setLineWidth(1);c.line(42,811,553,811);c.setFont('CV',7);c.setFillColor(colors.HexColor('#63778b'));c.drawString(42,24,'CarVision 2.12 TEST | Sofia Rakhmankulova | '+('technical report' if args.repository else 'technical draft'));c.drawRightString(553,24,str(doc.page)+' / 6')
    pdf=out/(args.output_stem+'.pdf')
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=45,bottomMargin=40,title='CarVision 2.12 - Midterm Technical Report',author='Sofia Rakhmankulova')
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
    (out/(args.output_stem+'.txt')).write_text('\n'.join(plain),encoding='utf-8-sig')
    from pypdf import PdfReader
    assert len(PdfReader(str(pdf)).pages)==6,'Report must be exactly six pages; check overflow'
    print(json.dumps(dict(pdf=str(pdf),pages=6,validation=stats),indent=2),flush=True)

if __name__=='__main__':main()
