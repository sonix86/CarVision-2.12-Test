"""Offline desktop studio; examples are images, never a table of answers."""
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk,filedialog,messagebox
from PIL import ImageTk,Image,ImageDraw
from .core import ROOT,BRANDS,COLOURS,Pipeline,read_image,save_prediction,annotation_brands
from .overlays import marker_positions,instance_colour
from .vehicle_model import vehicle_name,detailed_answer
from .version import BUILD_VERSION

BG='#f4f7fb';INK='#203249';MUTED='#738297';TEAL='#087f8c';NAVY='#17283e';CARD='#ffffff'
PAINT={'white':'#f5f5f2','black':'#242a31','grey':'#9ca7b2','red':'#d85659','blue':'#4b84c4','green':'#59a382','brown':'#a18364','yellow':'#edc459'}
MODES={'Обычный поиск':'fast','Мелкие машины · дольше':'balanced','Тщательный поиск · дольше':'detail'}
SCOPES=['Весь кадр','Машины в области','Одна машина в области']
COLOUR_RU=dict(zip(COLOURS,['Белый','Чёрный','Серый / серебро','Красный','Синий','Зелёный','Коричневый','Жёлтый']))

class App:
    def __init__(self,root):
        self.root=root;root.title('CarVision · София Рахманкулова');root.geometry('1320x860');root.minsize(1120,760)
        self.events=queue.Queue();self.pipeline=None;self.image=None;self.path=None;self.result=None;self.busy=False;self.process=None;self.closed=False;self.cancelled=False
        self.label_image=None;self.box=None;self.start=None;self.label_rows=[];self.undo_rows=[];self.transforms={};self.resize_job=None;self.buttons={}
        self.region=None;self.region_start=None;self.zoom_window=None;self.before_merge=None
        style=ttk.Style();style.theme_use('clam');style.configure('.',font=('Segoe UI',10))
        style.configure('TFrame',background=BG);style.configure('TLabel',background=BG,foreground=INK)
        style.configure('Title.TLabel',font=('Segoe UI',24,'bold'));style.configure('Muted.TLabel',foreground=MUTED)
        style.configure('TButton',padding=(12,9),borderwidth=0,background='#e7edf4',foreground=INK,focusthickness=0)
        style.map('TButton',background=[('active','#d8e4ee')],foreground=[('disabled','#98a6b7')])
        style.configure('TCheckbutton',background=BG,foreground=MUTED,padding=4)
        style.map('TCheckbutton',background=[('active',BG)])
        style.configure('Card.TFrame',background=CARD)
        style.configure('Card.TLabel',background=CARD,foreground=INK)
        style.configure('CardMuted.TLabel',background=CARD,foreground=MUTED)
        style.configure('Evidence.Horizontal.TProgressbar',background='#51b9b0',troughcolor='#edf2f6',borderwidth=0,lightcolor='#51b9b0',darkcolor='#51b9b0',thickness=4)
        style.configure('TProgressbar',background=TEAL,troughcolor='#e7edf4',borderwidth=0)
        style.configure('Primary.TButton',background=TEAL,foreground='white');style.map('Primary.TButton',background=[('active','#066774'),('disabled','#bdd3d6')])
        style.configure('Treeview',rowheight=32,font=('Segoe UI',10),background='white',fieldbackground='white');style.configure('Treeview.Heading',background='#e4eaf2',font=('Segoe UI',9,'bold'))
        style.map('Treeview',background=[('selected','#cbece6')],foreground=[('selected',INK)])
        rail=tk.Frame(root,bg=NAVY,width=190);rail.pack(side='left',fill='y');rail.pack_propagate(False)
        tk.Label(rail,text='CV',font=('Segoe UI',32,'bold'),fg='#6ed9c8',bg=NAVY).pack(anchor='w',padx=22,pady=(28,2))
        tk.Label(rail,text='CarVision',font=('Segoe UI',18,'bold'),fg='white',bg=NAVY,justify='left').pack(anchor='w',padx=22)
        version_file=ROOT/'models/model_version.json'
        version=json.loads(version_file.read_text(encoding='utf-8')).get('version','2.6') if version_file.exists() else '2.6'
        root.title('CarVision '+BUILD_VERSION+' · София Рахманкулова')
        tk.Label(rail,text='ТАШКЕНТ  /  '+BUILD_VERSION,font=('Segoe UI',8),fg='#a4b7cd',bg=NAVY).pack(anchor='w',padx=23,pady=(8,9))
        if str(version)!=BUILD_VERSION:tk.Label(rail,text='Набор весов: '+str(version),font=('Segoe UI',8),fg='#f2c36c',bg=NAVY).pack(anchor='w',padx=23)
        tk.Label(rail,text='●  Работает без интернета',font=('Segoe UI',8),fg='#7dd5bf',bg=NAVY).pack(anchor='w',padx=22,pady=(0,29))
        self.nav={};self.views={}
        for key,title in [('infer','Анализ фото'),('labels','Учебная разметка'),('training','Обучение'),('help','О проекте')]:
            b=tk.Button(rail,text=title,command=lambda k=key:self.show(k),bg=NAVY,fg='#bbcadb',activebackground='#26445e',activeforeground='white',font=('Segoe UI',11),anchor='w',relief='flat',padx=18,pady=14,cursor='hand2');b.pack(fill='x',padx=10,pady=3);self.nav[key]=b
        tk.Label(rail,text='Владелец проекта\nСофия Рахманкулова',font=('Segoe UI',10),fg='#c4d3e4',bg=NAVY,justify='left').pack(side='bottom',anchor='w',padx=20,pady=22)
        main=ttk.Frame(root,padding=(24,18));main.pack(side='left',fill='both',expand=True)
        self.title=tk.StringVar();ttk.Label(main,textvariable=self.title,style='Title.TLabel').pack(anchor='w')
        ttk.Label(main,text='Сначала марка. Модель — когда достаточно деталей. Цвет — отдельно.',style='Muted.TLabel').pack(anchor='w',pady=(5,18))
        container=ttk.Frame(main);container.pack(fill='both',expand=True)
        for key in self.nav:self.views[key]=ttk.Frame(container)
        self.build_infer();self.build_labels();self.build_training();self.build_help()
        bottom=ttk.Frame(main);bottom.pack(fill='x',pady=(12,0));self.status=tk.StringVar(value='Готово. Добавьте фото или начните разметку своего набора.')
        ttk.Label(bottom,textvariable=self.status,style='Muted.TLabel',wraplength=850).pack(side='left',fill='x',expand=True)
        self.progress=ttk.Progressbar(bottom,mode='indeterminate',length=80);self.progress.pack(side='right')
        root.bind('<Control-o>',lambda e:self.open_infer());root.bind('<Control-Return>',lambda e:self.predict());root.protocol('WM_DELETE_WINDOW',self.close)
        self.show('infer');self.refresh_buttons();root.after(80,self.poll)

    def button(self,parent,key,text,callback,primary=False,**pack):
        b=ttk.Button(parent,text=text,command=callback,style='Primary.TButton' if primary else 'TButton');b.pack(**pack);self.buttons[key]=b;return b
    def show(self,key):
        for name,view in self.views.items():view.pack_forget();self.nav[name].configure(bg='#26445e' if key==name else NAVY)
        self.views[key].pack(fill='both',expand=True);self.current=key;self.title.set(dict(infer='Анализ фотографии',labels='Разметка для обучения',training='Обучение на своих данных',help='О проекте')[key]);self.root.after_idle(self.redraw)
        if key=='training':
            rows=self.all_labels();self.training_data_summary.set(f"Ваш набор: {len(rows)} машин · {len({r['image'] for r in rows})} фото · {len({r['brand'] for r in rows})} марок · {len({r['group'] for r in rows})} групп съёмки")

    def build_infer(self):
        view=self.views['infer'];bar=ttk.Frame(view);bar.pack(fill='x')
        self.button(bar,'open','Открыть фото',self.open_infer,True,side='left')
        self.button(bar,'samples','Примеры фото',lambda:self.open_infer(ROOT/'samples'),side='left',padx=7)
        self.button(bar,'choose_region','Выбрать зону мышью',self.choose_region,side='left')
        self.button(bar,'clear_region','Вся фотография',self.clear_region,side='left',padx=7)
        self.scope=tk.StringVar(value=SCOPES[0]);self.region_hint=tk.StringVar(value='Обведите нужную машину или распознайте всё фото. Цветные контуры различают машины; цвет кузова указан справа.')
        ttk.Label(view,textvariable=self.region_hint,style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(7,0))
        actions=ttk.Frame(view);actions.pack(fill='x',pady=8)
        self.button(actions,'zoom','Крупнее / выбранная машина',self.zoom_car,side='left')
        self.advanced=tk.BooleanVar(value=False);ttk.Checkbutton(actions,text='Подробные настройки',variable=self.advanced,command=self.toggle_advanced).pack(side='left',padx=12)
        self.button(actions,'predict','Анализировать фото',self.predict,True,side='right')
        self.advanced_controls=ttk.Frame(view,padding=(10,7))
        line=ttk.Frame(self.advanced_controls);line.pack(fill='x')
        ttk.Label(line,text='Где искать:').pack(side='left');self.scope_choice=ttk.Combobox(line,textvariable=self.scope,values=SCOPES,state='readonly',width=25);self.scope_choice.pack(side='left',padx=7);self.scope_choice.bind('<<ComboboxSelected>>',lambda e:self.scope_changed())
        self.mode=tk.StringVar(value='Обычный поиск');self.mode_choice=ttk.Combobox(line,textvariable=self.mode,values=list(MODES),state='readonly',width=27);self.mode_choice.pack(side='left',padx=7)
        scores=ttk.Frame(self.advanced_controls);scores.pack(fill='x',pady=(5,0));self.det=tk.DoubleVar(value=.5);self.conf=tk.DoubleVar(value=.5)
        for text,var in [('Уверенность поиска',self.det),('Уверенность марки',self.conf)]:
            ttk.Label(scores,text=text).pack(side='left',padx=(0,5));ttk.Spinbox(scores,from_=.1,to=.99,increment=.05,textvariable=var,width=5).pack(side='left',padx=(0,15))
        ttk.Label(self.advanced_controls,text='Поиск: больше значение — меньше сомнительных объектов. Марка: ниже выбранного значения показываются варианты вместо установленной марки. Значения 0.5 подходят для начала; проценты сети не гарантируют правильность.',wraplength=800,style='Muted.TLabel',font=('Segoe UI',9)).pack(anchor='w',pady=(4,0))
        corrections=ttk.Frame(self.advanced_controls);corrections.pack(fill='x',pady=(5,0))
        self.button(corrections,'merge_parts','Объединить части одной машины',self.choose_parts,side='left').configure(padding=(8,5))
        self.button(corrections,'undo_merge','Отменить объединение',self.undo_merge,side='left',padx=7).configure(padding=(8,5))
        self.model_status=tk.StringVar();ttk.Label(view,textvariable=self.model_status,foreground='#a66718',wraplength=850).pack(anchor='w',pady=(0,7))
        body=ttk.Frame(view);body.pack(fill='both',expand=True);self.infer_body=body
        self.preview=tk.Canvas(body,bg='#eaf0f6',highlightthickness=1,highlightbackground='#dfe7ef');self.preview.pack(side='left',fill='both',expand=True)
        panel=ttk.Frame(body,padding=(14,0,0,0),width=320);panel.pack(side='right',fill='y');panel.pack_propagate(False)
        self.results_view=tk.Canvas(panel,bg=BG,highlightthickness=0,width=285);self.results_view.pack(side='left',fill='both',expand=True)
        panel_scroll=ttk.Scrollbar(panel,command=self.results_view.yview);panel_scroll.pack(side='right',fill='y');self.results_view.configure(yscrollcommand=panel_scroll.set)
        right=ttk.Frame(self.results_view);right_window=self.results_view.create_window(0,0,window=right,anchor='nw')
        right.bind('<Configure>',lambda e:self.results_view.configure(scrollregion=self.results_view.bbox('all')))
        self.results_view.bind('<Configure>',lambda e:self.results_view.itemconfigure(right_window,width=e.width))
        self.results_view.bind('<MouseWheel>',lambda e:self.results_view.yview_scroll(-int(e.delta/120),'units'))
        ttk.Label(right,text='Найденные машины',font=('Segoe UI',13,'bold')).pack(anchor='w');self.summary=tk.StringVar(value='Добавьте фото, чтобы начать');ttk.Label(right,textvariable=self.summary,style='Muted.TLabel',wraplength=275).pack(anchor='w',pady=(5,8))
        tablebox=ttk.Frame(right);tablebox.pack(fill='x');self.table=ttk.Treeview(tablebox,columns=('id','brand','colour'),show='headings',height=3,selectmode='browse')
        for name,title,width in [('id','№',30),('brand','Автомобиль / вариант',160),('colour','Цвет',90)]:self.table.heading(name,text=title);self.table.column(name,width=width,minwidth=25,stretch=name!='id')
        self.table.pack(side='left',fill='x',expand=True);scroll=ttk.Scrollbar(tablebox,command=self.table.yview);scroll.pack(side='right',fill='y');self.table.configure(yscrollcommand=scroll.set);self.table.bind('<<TreeviewSelect>>',lambda e:self.redraw())
        self.evidence_card=ttk.Frame(right,style='Card.TFrame',padding=12);self.evidence_card.pack(fill='x',pady=(10,0))
        self.selected_title=tk.StringVar(value='Выберите машину')
        ttk.Label(self.evidence_card,textvariable=self.selected_title,style='Card.TLabel',font=('Segoe UI',11,'bold'),wraplength=248).pack(anchor='w')
        self.colour_chip=tk.Label(self.evidence_card,text='  Цвет появится после анализа',bg=CARD,fg=MUTED,font=('Segoe UI',9),anchor='w',padx=5,pady=4);self.colour_chip.pack(fill='x',pady=(6,9))
        self.variants_caption=tk.StringVar(value='Варианты распознавания')
        ttk.Label(self.evidence_card,textvariable=self.variants_caption,style='CardMuted.TLabel',font=('Segoe UI',9)).pack(anchor='w',pady=(0,4))
        self.variant_rows=[]
        for _ in range(3):
            label=ttk.Label(self.evidence_card,text='—',style='Card.TLabel',font=('Segoe UI',9),wraplength=248);label.pack(anchor='w',fill='x')
            meter=ttk.Progressbar(self.evidence_card,style='Evidence.Horizontal.TProgressbar',maximum=1);meter.pack(fill='x',pady=(3,7));self.variant_rows.append((label,meter))
        self.button(self.evidence_card,'ranking','Все варианты марки',self.show_model_ranking,fill='x',pady=(0,6)).configure(padding=(7,5))
        self.details=tk.StringVar(value='Нажмите на контур или строку результата. Варианты — предположения сети, их можно проверить на крупном изображении.')
        ttk.Label(self.evidence_card,textvariable=self.details,style='CardMuted.TLabel',font=('Segoe UI',9),wraplength=248).pack(anchor='w',pady=(3,0))
        exports=ttk.Frame(view);exports.pack(fill='x',pady=(8,0))
        for key,text,fn in [('copy','Ответ для задания',self.copy),('copy_details','Копировать варианты',self.copy_details),('export','Сохранить результаты',self.export),('output','Открыть папку',lambda:self.open_folder(ROOT/'outputs'))]:self.button(exports,key,text,fn,side='left',padx=(0,7)).configure(padding=(9,6))
        self.preview.bind('<Configure>',lambda e:self.queue_redraw());self.preview.bind('<ButtonPress-1>',self.start_region);self.preview.bind('<B1-Motion>',self.drag_region);self.preview.bind('<ButtonRelease-1>',self.end_region);self.preview.bind('<Double-Button-1>',lambda e:self.zoom_car());self.table.bind('<Double-Button-1>',lambda e:self.zoom_car())

    def toggle_advanced(self):
        if self.advanced.get():self.advanced_controls.pack(before=self.infer_body,fill='x',pady=(0,7))
        else:self.advanced_controls.pack_forget()
        self.queue_redraw()

    def choose_parts(self):
        if self.busy or not self.result or len(self.result['cars'])<2:return
        win=tk.Toplevel(self.root);win.title('Части одной машины');win.geometry('480x490');win.transient(self.root);win.grab_set()
        ttk.Label(win,text='Отметьте контуры, которые принадлежат ОДНОЙ машине. Их видимые пиксели объединятся, марка, модель и цвет будут рассчитаны заново. Пространство между частями останется исключённым.',wraplength=435,padding=14).pack(anchor='w')
        canvas=tk.Canvas(win,bg=BG,highlightthickness=0);canvas.pack(fill='both',expand=True);bar=ttk.Scrollbar(canvas,command=canvas.yview);bar.pack(side='right',fill='y');canvas.configure(yscrollcommand=bar.set)
        body=ttk.Frame(canvas);canvas.create_window(0,0,window=body,anchor='nw');body.bind('<Configure>',lambda e:canvas.configure(scrollregion=canvas.bbox('all')))
        choices=[]
        for row in self.result['cars']:
            var=tk.BooleanVar(value=False);choices.append((row['id'],var));ttk.Checkbutton(body,text=f"№ {row['id']} · {vehicle_name(row)}",variable=var).pack(anchor='w',padx=14,pady=4)
        def apply():
            ids=[i for i,var in choices if var.get()]
            if len(ids)<2:messagebox.showinfo('Выбор частей','Отметьте минимум два контура.',parent=win);return
            self.before_merge=self.result;win.destroy();self.set_busy(True,'Объединяю видимые части и пересчитываю ответ…')
            def work():
                try:
                    from .scene import merge_prediction
                    result=merge_prediction(self.pipeline,self.image,self.result,ids);save_prediction(self.path,result,ROOT/'outputs');self.events.put(('prediction',result))
                except Exception as e:self.events.put(('error',str(e)))
            threading.Thread(target=work,daemon=True).start()
        ttk.Button(win,text='Объединить и пересчитать',command=apply,style='Primary.TButton').pack(fill='x',padx=14,pady=14)

    def undo_merge(self):
        if self.busy or self.before_merge is None:return
        result=self.before_merge;self.before_merge=None;self.events.put(('prediction',result));self.status.set('Объединение отменено.')

    def choose_region(self):
        if self.busy or self.image is None:return
        self.scope.set(SCOPES[2]);self.region=None;self.scope_changed();self.status.set('Протяните мышью от одного угла нужной области до другого. Затем нажмите «Распознать в зоне».')

    def result_details(self,row,include_variants=True):
        evidence=row.get('make_evidence');make_options=bool(evidence and not evidence['accepted'])
        variants=evidence['alternatives'] if make_options else row.get('model_alternatives',[])
        text=('Возможные марки:\n'+'\n'.join(f"{r.get('display_name',r['brand'])} — {r['confidence']:.0%}" for r in variants) if make_options else
              'Возможные модели:\n'+'\n'.join(f"{r['brand']} {r['model']} — {r['confidence']:.0%}" for r in variants) if variants else 'Модель пока не определена.')
        if not include_variants:text=''
        text+=('\n' if text else '')+'Марка: '+('не установлена' if row['brand']=='unknown' else row['brand']+f" · {row['brand_confidence']:.0%}")
        if row.get('make_evidence'):
            evidence=row['make_evidence']
            text+='\nМарка и модель оцениваются отдельно.'
            if not evidence['accepted']:text+='\nПервый вариант марки: '+evidence['candidate']+f" · {evidence['confidence']:.0%}."
            if evidence.get('acceptance_reason'):text+='\n'+evidence['acceptance_reason']
        if row.get('brand_model_conflict'):
            text+='\nЕсть противоречие: отдельная сеть марки предлагает '+row['classifier_brand']+'; сеть моделей — '+row.get('model_candidate_brand','?')+'. Название с ? — первый вариант, не подтверждённый ответ.'
        colours=row.get('colour_alternatives',[])
        text+='\nЦвет: '+COLOUR_RU.get(row['colour'],row['colour'])+f" · {row.get('colour_confidence',0):.0%}"+(' · проверьте' if not row.get('colour_reliable',True) else '')
        if len(colours)>1:text+='\nЕщё: '+COLOUR_RU.get(colours[1]['colour'],colours[1]['colour'])+f" · {colours[1]['confidence']:.0%}"
        from .vehicle_model import model_status_text
        model_status=model_status_text(row)
        if model_status:text+='\n'+model_status
        badge=row.get('badge_evidence')
        if badge and badge.get('accepted'):text+='\nЭмблема: '+badge['brand']+' — дополнительный признак, не подтверждение конкретной модели.'
        elif badge:text+='\nЭмблема не подтверждена: решение по кузову и деталям.'
        if row.get('brand_source','').startswith('own model head'):text+='\nМарка установлена обученным классификатором моделей кузова; её нет в прежнем отдельном классификаторе марок.'
        for reason in row.get('visual_limitations',[]):text+='\n'+reason
        if row.get('reference_comparison'):text+='\nСравнение с целыми машинами и фрагментами учебного набора. Проценты относительны; неизвестные модели могут быть похожи.'
        if row.get('approximate_mask'):text+='\nГраница приблизительная; наличие машины не подтверждено.'
        elif row.get('segmentation'):text+='\nРаспознана видимая форма машины.'
        if row.get('merged_proposals',1)>1:text+=f"\nОбъединено предложений детектора: {row['merged_proposals']}. Все части с этим номером относятся к одной машине; скрытые пиксели не дорисованы."
        if row.get('visible_parts',1)>1:text+=f"\nВидимых участков кузова: {row['visible_parts']}. Цвет контура служит только для различения машин."
        return text

    def show_evidence(self,row=None):
        self.selected_title.set(vehicle_name(row) if row else 'Выберите машину')
        evidence=row.get('make_evidence') if row else None
        make_options=bool(evidence and not evidence['accepted'])
        options=evidence['alternatives'] if make_options else row.get('model_alternatives',[]) if row else []
        self.variants_caption.set('Возможные марки' if make_options else 'Варианты модели этой марки' if row and row['brand']!='unknown' else 'Варианты распознавания')
        for (label,meter),variant in zip(self.variant_rows,options+[None]*3):
            name=((variant.get('display_name',variant['brand']) if make_options else variant['brand']+' '+variant.get('model',''))).strip() if variant else ''
            label.configure(text=f"{name} · {variant['confidence']:.0%}" if variant else '—');meter['value']=variant['confidence'] if variant else 0
        if row:
            colour=COLOUR_RU.get(row['colour'],row['colour']);uncertain=not row.get('colour_reliable',True)
            self.colour_chip.configure(text=('≈ ' if uncertain else '')+colour,bg=PAINT.get(row['colour'],CARD),fg='white' if row['colour'] in ('black','blue','red','green','brown') else INK)
            self.details.set(self.result_details(row,include_variants=False))
        else:
            self.colour_chip.configure(text='Цвет появится после анализа',bg=CARD,fg=MUTED)
            self.details.set('Нажмите на контур или строку. Проценты сети не гарантируют правильность; неизвестная модель может получить похожий вариант.')

    def show_model_ranking(self):
        selection=self.table.selection()
        if not selection or not self.result:return
        row=next(r for r in self.result['cars'] if r['id']==int(selection[0]));make_mode=bool(row.get('make_evidence'))
        ranking=row['make_evidence'].get('ranking',row['make_evidence']['alternatives']) if make_mode else row.get('model_ranking',row.get('model_alternatives',[]))
        window=tk.Toplevel(self.root);window.title('Варианты марки' if make_mode else 'Сравнение видимых деталей');window.geometry('670x600')
        frame=ttk.Frame(window,padding=20);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text=f"{'Изученные марки' if make_mode else 'Изученные варианты'}: {len(ranking)}",font=('Segoe UI',17,'bold')).pack(anchor='w')
        ttk.Label(frame,text='Это ранжирование обученных классов, а не список всех существующих автомобилей. Одинаковая крыша или фара может встречаться у разных моделей. Уверенный вариант ещё не доказывает, что машина известна сети.',style='Muted.TLabel',wraplength=620).pack(anchor='w',pady=10)
        table=ttk.Treeview(frame,columns=('rank','model','score'),show='headings',height=12)
        for key,title,width in [('rank','№',40),('model','Марка' if make_mode else 'Марка и модель',350),('score','Сходство',100)]:table.heading(key,text=title);table.column(key,width=width)
        table.pack(fill='both',expand=True)
        for i,r in enumerate(ranking,1):table.insert('','end',values=(i,(r.get('display_name',r['brand'])+' '+r.get('model','')).strip(),f"{r['confidence']:.1%}"))
        ttk.Label(frame,text='\n'.join(row.get('visual_limitations',[])) or 'Для проверки загрузите более крупный кадр машины с другого ракурса.',style='Muted.TLabel',wraplength=620).pack(anchor='w',pady=(10,0))

    def build_labels(self):
        view=self.views['labels'];ttk.Label(view,text='Подготовка учебных ответов. В «Анализе фото» программа определяет марку и цвет автоматически.',style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(0,10));bar=ttk.Frame(view);bar.pack(fill='x')
        self.button(bar,'label_open','Добавить своё фото',self.open_label,True,side='left');self.button(bar,'label_save','Сохранить машину',self.save_label,side='left',padx=7);self.button(bar,'undo','Отменить последнюю',self.undo_label,side='left')
        fields=ttk.Frame(view);fields.pack(fill='x',pady=12);self.brand=tk.StringVar(value=BRANDS[0]);self.colour=tk.StringVar(value=COLOURS[0]);self.group=tk.StringVar();self.source=tk.StringVar(value='own');self.keep_group=tk.BooleanVar()
        for title,var,values in [('Марка',self.brand,annotation_brands()),('Цвет',self.colour,COLOURS)]:ttk.Label(fields,text=title).pack(side='left',padx=(0,6));ttk.Combobox(fields,textvariable=var,values=values,state='normal' if title=='Марка' else 'readonly',width=18).pack(side='left',padx=(0,18))
        meta=ttk.Frame(view);meta.pack(fill='x',pady=(0,10))
        for title,var in [('Группа / сессия',self.group),('Источник',self.source)]:ttk.Label(meta,text=title,style='Muted.TLabel').pack(side='left',padx=(0,6));ttk.Entry(meta,textvariable=var,width=21).pack(side='left',padx=(0,14))
        ttk.Checkbutton(meta,text='Одна группа для серии',variable=self.keep_group).pack(side='left')
        self.canvas=tk.Canvas(view,bg='#e4eaf2',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        for event,fn in [('<ButtonPress-1>',self.start_box),('<B1-Motion>',self.drag_box),('<ButtonRelease-1>',self.end_box)]:self.canvas.bind(event,fn)
        self.canvas.bind('<Motion>',self.hover_label)
        self.canvas.bind('<Configure>',lambda e:self.queue_redraw());self.label_status=tk.StringVar(value='Обведите каждую машину. Зелёные рамки сохранены, жёлтая рамка — новая.');ttk.Label(view,textvariable=self.label_status,style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(10,0))

    def build_training(self):
        view=self.views['training'];ttk.Label(view,text='Научить программу на своих фотографиях',font=('Segoe UI',14,'bold'),wraplength=850).pack(anchor='w',pady=(0,8))
        self.training_guide='Для распознавания уже есть обученные веса — просто откройте «Анализ фото». Этот раздел нужен для собственного набора Ташкента по заданию. Загрузка фото на анализ сама по себе ничего не обучает.\n\n1. Учебная разметка: добавьте свои снимки, обведите машины и укажите правильные марки и цвета. Серию одной машины сохраняйте одной группой, чтобы она не попала одновременно в обучение и проверку.\n2. Проверить данные: проверяет файлы, координаты, подписи и независимость групп. Нужны как минимум две марки в обучении и две независимые группы; для качества требуется значительно больше.\n3. Обучить: обновляет классификатор марок/цветов и резервный детектор рамок. Эпоха — один проход по учебным данным; больше проходов не всегда лучше. Старые веса сохраняются в models/backups.\n4. Проверка качества: показывает ошибки на сохранённой проверочной части. Затем проверьте новые фото, которых не было в обучении.\n\nКонтурный COCO-сегментатор здесь не переобучается. Конкретные модели (Cobalt, Spark и другие) и сравнение фрагментов обучаются отдельными рецептами source/tools; эта кнопка не добавляет имена новых моделей.'
        self.training_guide+='\n\nУчебная версия сохраняется отдельно в models/own_training_candidate. Рабочее распознавание не заменяется автоматически: сначала сравните качество. «Проверка качества» проверяет именно эту учебную версию. Полученные10 фото и частичная разметка лежат отдельно в data/own_oct21; одной поездки недостаточно для независимых train/validation.'
        self.training_guide+='\n\nРасширенный классификатор semantic_make.pt использует общий визуальный экстрактор CLIP и собственную голову марок, обученную на размеченных изображениях машин. Подготовка, обучение и проверка описаны в tools/*v26.py и Development/EV26. Учебная кнопка тренирует отдельную учебную архитектуру и не заменяет расширенный классификатор. Подсказки марок в разметке — справочник; само наличие названия не означает обученное распознавание.'
        ttk.Label(view,text=self.training_guide,style='Muted.TLabel',wraplength=850,justify='left',font=('Segoe UI',9)).pack(anchor='w',pady=(0,10))
        self.training_data_summary=tk.StringVar();ttk.Label(view,textvariable=self.training_data_summary,font=('Segoe UI',10,'bold')).pack(anchor='w',pady=(0,10))
        fields=ttk.Frame(view);fields.pack(fill='x');self.epochs=tk.IntVar(value=25);self.detector_epochs=tk.IntVar(value=25)
        for title,var in [('Эпохи классификатора',self.epochs),('Эпохи детектора',self.detector_epochs)]:ttk.Label(fields,text=title).pack(side='left',padx=(0,7));ttk.Spinbox(fields,from_=1,to=1000,textvariable=var,width=6).pack(side='left',padx=(0,20))
        bar=ttk.Frame(view);bar.pack(fill='x',pady=12)
        self.button(bar,'audit','Проверить данные',lambda:self.command(['audit','--labels',str(ROOT/'data/labels.jsonl')]),side='left');self.button(bar,'train','Обучить учебного кандидата',self.train,True,side='left',padx=7);self.button(bar,'evaluate','Проверка качества',self.evaluate,side='left');self.button(bar,'cancel','Остановить',self.cancel,side='right')
        self.log=tk.Text(view,bg=NAVY,fg='#d6e4ef',font=('Consolas',10),relief='flat',wrap='word',padx=14,pady=14,state='disabled');self.log.pack(fill='both',expand=True);self.append_log('Обучение локально, без API. Общие веса уже включены в пакет.\nГотовые классификаторы марок и цветов не используются.\nОценка на вырезках не заменяет проверку полного конвейера на новых фото.\n')

    def build_help(self):
        view=self.views['help'];ttk.Label(view,text='София Рахманкулова',font=('Segoe UI',15,'bold')).pack(anchor='w',pady=(0,15))
        text='Откройте фотографию во вкладке «Анализ фото» и нажмите «Анализировать». Марку выбирать не нужно. Сначала программа оценивает марку отдельной обученной сетью. Если уверенности достаточно, марка сохраняется даже тогда, когда конкретная модель неизвестна. Если не уверена марка, показываются варианты марок.\n\nМодель и цвет оцениваются отдельно. Не все модели автомобилей изучены: уверенное определение BYD не означает уверенное определение Seal или Song Plus. Для маленькой или перекрытой машины выделите область мышью и попробуйте крупный просмотр. Неизвестная сетью машина тоже может получить ошибочный похожий вариант.\n\nСписок изученных марок, объём данных и результаты проверки находятся в инструкции обновления. Справочник в учебной разметке служит для подписи своих фотографий, а не подтверждает качество распознавания. Загрузка фото на анализ сама по себе не обучает сеть.\n\nДля задания добавляйте собственные снимки Ташкента, правильно подписывайте марки, цвета и группы съёмки. Для переноса копируйте всю папку CarVision, включая models и _internal. Программа работает локально без API и ключей.'
        ttk.Label(view,text=text,wraplength=850,justify='left').pack(anchor='w',pady=(0,18))
        for key,text,fn in [('readme','Полная инструкция',lambda:self.open_document(ROOT/'GUIDE_FULL_2_6_RU.txt' if (ROOT/'GUIDE_FULL_2_6_RU.txt').exists() else ROOT/'README.md')),('checklist','Список требований',lambda:self.open_document(ROOT/'CHECKLIST.md')),('help_pdf','PDF с примерами',self.open_pdf),('folder','Папка проекта',lambda:self.open_folder(ROOT))]:self.button(view,key,text,fn,anchor='w',pady=4)
        ttk.Label(view,text='Публичные примеры в samples не являются собственным обучающим набором Софии. Авторы, ссылки и лицензии сохранены. GitHub будет подключён после передачи ссылки владельцем.',style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=20)

    def refresh_buttons(self):
        detector=(ROOT/'models/detector.pt').exists();classifier=(ROOT/'models/classifier.pt').exists()
        self.scope_choice.configure(state='disabled' if self.busy else 'readonly');self.mode_choice.configure(state='disabled' if self.busy else 'readonly')
        for key,b in self.buttons.items():
            enabled=not self.busy
            if key=='choose_region':enabled=enabled and self.image is not None
            if key=='predict':enabled=enabled and self.path is not None and detector
            if key in ('copy','copy_details','export'):enabled=enabled and self.result is not None
            if key=='zoom':enabled=enabled and (self.region is not None or bool(self.table.selection()))
            if key=='ranking':enabled=enabled and self.result is not None and bool(self.table.selection())
            if key=='merge_parts':enabled=enabled and self.pipeline is not None and self.result is not None and len(self.result['cars'])>=2
            if key=='undo_merge':enabled=enabled and self.before_merge is not None
            if key=='clear_region':enabled=enabled and (self.region is not None or self.scope.get()!=SCOPES[0])
            if key=='label_save':enabled=enabled and self.box is not None
            if key=='undo':enabled=enabled and bool(self.undo_rows)
            if key=='cancel':enabled=self.busy and self.process is not None
            b.configure(state='normal' if enabled else 'disabled')
        self.buttons['predict'].configure(text='Распознать в зоне' if self.scope.get()!=SCOPES[0] else 'Анализировать фото')
        self.model_status.set('Готово. Выберите весь кадр или обведите нужную машину. Марка, модель и цвет определяются автоматически.' if detector and classifier else 'Обучение не завершено: '+('нужны детектор и классификатор.' if not detector and not classifier else 'нужен детектор.' if not detector else 'нужен классификатор.'))
        if detector and classifier and not any((ROOT/'models'/name).exists() for name in ('semantic_make.pt','make_expert.pt')):self.model_status.set('Расширенный классификатор марок отсутствует. Работает прежняя версия распознавания; установите обновление полностью.')
    def evaluate(self):
        if (ROOT/'models/evaluation/public_test.json').exists() and not self.all_labels():
            self.open_document(ROOT/'PROTOTYPE_VALIDATION.md')
        else:
            directory=ROOT/'models/own_training_candidate'
            if not (directory/'training.json').exists():
                messagebox.showinfo('Проверка учебной версии','Сначала завершите обучение на размеченном наборе. Результаты текущего выпуска находятся в VALIDATION.md.');return
            self.command(['evaluate','--labels',str(ROOT/'data/labels.jsonl'),'--models',str(directory),'--output',str(directory/'evaluation')])
    def set_busy(self,value,text):
        self.busy=value;self.status.set(text);self.progress.start(12) if value else self.progress.stop();self.refresh_buttons()
    def open_folder(self,path):
        try:path.mkdir(parents=True,exist_ok=True);os.startfile(path)
        except Exception as e:messagebox.showerror('Папка',str(e))
    def open_document(self,path):
        try:
            if not path.exists():raise FileNotFoundError('Файл отсутствует: '+str(path))
            os.startfile(path)
        except Exception as e:messagebox.showerror('Открытие файла',str(e))
    def open_pdf(self):self.open_document(ROOT/'samples/Tashkent_Photo_Examples.pdf')
    def open_infer(self,initialdir=None):
        if self.busy:return
        path=filedialog.askopenfilename(title='Фотография для анализа',initialdir=str(initialdir or ROOT),filetypes=[('Фотографии','*.jpg *.jpeg *.png *.bmp *.webp')])
        if path:self.load_photo(path)
    def load_photo(self,path):
        try:image=read_image(path)
        except Exception as e:messagebox.showerror('Ошибка фото',str(e));return
        if self.zoom_window is not None and self.zoom_window.winfo_exists():self.zoom_window.destroy()
        self.before_merge=None;self.path=str(path);self.image=image;self.result=None;self.region=None;self.region_start=None;self.table.delete(*self.table.get_children());self.show_evidence();self.scope.set(SCOPES[0]);self.scope_changed()
        self.summary.set(f'{Path(path).name}\n{image.width} × {image.height} пикселей');self.show('infer');self.status.set('Фото загружено. '+('Можно анализировать.' if (ROOT/'models/detector.pt').exists() else 'Сначала обучите детектор на своей разметке.'));self.refresh_buttons();self.redraw()
    def queue_redraw(self):
        if self.closed:return
        if self.resize_job:
            try:self.root.after_cancel(self.resize_job)
            except tk.TclError:pass
        self.resize_job=self.root.after(80,self.redraw)
    def draw(self,canvas,image,rows=None,selected=None,enlarge=False):
        canvas.delete('all');w=max(1,canvas.winfo_width());h=max(1,canvas.winfo_height())
        if image is None:canvas.create_text(w/2,h/2-15,text='Добавьте фотографию',fill=MUTED,font=('Segoe UI',18,'bold'));canvas.create_text(w/2,h/2+22,text='JPEG · PNG · BMP · WebP',fill='#8897a9',font=('Segoe UI',11));return
        im=image.copy()
        if enlarge:
            scale=min(max(1,w-24)/image.width,max(1,h-24)/image.height,8);im=im.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS)
        else:im.thumbnail((max(1,w-24),max(1,h-24)))
        ox=(w-im.width)//2;oy=(h-im.height)//2
        if any(row.get('contours') for row in rows or []):
            layer=Image.new('RGBA',im.size,(0,0,0,0));paint=ImageDraw.Draw(layer)
            for row in rows or []:
                from PIL import ImageColor
                tint=(255,179,65,46) if row.get('id')==selected else (*ImageColor.getrgb(instance_colour(row.get('id',1))),25)
                if row.get('segmentation'):
                    from .segmentation import decode_mask
                    encoded=row['segmentation'];mask=Image.fromarray(decode_mask(encoded).astype('uint8')*tint[3]);scale=encoded.get('scale',[1,1]);size=(max(1,round(mask.width*scale[0]*im.width/image.width)),max(1,round(mask.height*scale[1]*im.height/image.height)))
                    patch=Image.new('RGBA',size,tint);patch.putalpha(mask.resize(size,Image.Resampling.NEAREST));position=(round(encoded['origin'][0]*im.width/image.width),round(encoded['origin'][1]*im.height/image.height));layer.alpha_composite(patch,position)
                else:
                    for contour in row.get('contours',[]):paint.polygon([(x*im.width/image.width,y*im.height/image.height) for x,y in contour],fill=tint)
            im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')
        photo=ImageTk.PhotoImage(im);canvas.photo=photo;canvas.create_image(ox,oy,image=photo,anchor='nw')
        sx,sy=im.width/image.width,im.height/image.height;self.transforms[canvas]=(sx,sy,ox,oy)
        screen_boxes=[]
        for i,row in enumerate(rows or [],1):
            x,y,x2,y2=row['box'];coords=(ox+x*sx,oy+y*sy,ox+x2*sx,oy+y2*sy);ident=row.get('id',i)
            colour='#ffb341' if ident==selected else instance_colour(ident)
            if row.get('contours'):
                for contour in row['contours']:canvas.create_polygon(*[v for x,y in contour for v in (ox+x*sx,oy+y*sy)],outline=colour,fill='',width=3 if ident==selected else 2,tags=f'car-{ident}')
            else:canvas.create_rectangle(*coords,outline=colour,width=3 if ident==selected else 2)
            screen_boxes.append(coords)
        for i,(row,rect) in enumerate(zip(rows or [],marker_positions(screen_boxes,w,h)),1):
            ident=row.get('id',i);x,y,x2,y2=rect;box=screen_boxes[i-1];colour='#ffb341' if ident==selected else instance_colour(ident);canvas.create_line(box[0],box[1],(x+x2)/2,(y+y2)/2,fill=colour);canvas.create_rectangle(*rect,fill=TEAL if ident==selected else NAVY,outline=colour);canvas.create_text((x+x2)/2,(y+y2)/2,text=str(ident),fill='white',font=('Segoe UI',9,'bold'),tags='car-marker')
            tag=f'car-{ident}';canvas.addtag_withtag(tag,canvas.find_all()[-1]);canvas.addtag_withtag(tag,canvas.find_all()[-2])

    def select_car(self,ident):
        if not self.busy and self.scope.get()==SCOPES[0] and self.table.exists(str(ident)):
            self.table.selection_set(str(ident));self.table.see(str(ident));self.redraw()
    def redraw(self):
        self.resize_job=None
        if self.closed:return
        if self.current=='infer':
            selection=self.table.selection();selected=int(selection[0]) if selection else None;self.draw(self.preview,self.image,self.result['cars'] if self.result else [],selected)
            self.draw_region();self.refresh_buttons()
            if selected and self.result:
                row=next(r for r in self.result['cars'] if r['id']==selected);self.show_evidence(row)
        elif self.current=='labels':self.draw(self.canvas,self.label_image,self.label_rows);self.draw_selection()
    def predict(self):
        if self.busy or not self.path:return
        try:
            d,b=self.det.get(),self.conf.get();mode=MODES[self.mode.get()]
            if not 0<d<1 or not 0<b<1:raise ValueError('Пороги должны быть между 0 и 1.')
        except Exception as e:messagebox.showerror('Настройки',str(e));return
        region=self.region.copy() if self.region is not None and self.scope.get()!=SCOPES[0] else None;single=self.scope.get()==SCOPES[2]
        if self.scope.get()!=SCOPES[0] and region is None:messagebox.showinfo('Выберите область','Обведите мышью машину или участок фотографии.');return
        self.before_merge=None;source=self.path;self.set_busy(True,'Анализ выделенной области…' if region else 'Анализ фотографии…')
        def work():
            try:
                if self.pipeline is None:self.pipeline=Pipeline()
                self.pipeline.detection=d;self.pipeline.brand_threshold=b;self.pipeline.mode=mode;result=self.pipeline.predict(source,region=region,single=single);save_prediction(source,result,ROOT/'outputs');self.events.put(('prediction',result))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()
    def scope_changed(self):
        self.preview.configure(cursor='arrow' if self.scope.get()==SCOPES[0] else 'crosshair')
        self.region_hint.set('Нажмите на видимую форму машины или строку результата. В деталях — три вероятные модели.' if self.scope.get()==SCOPES[0] else 'Обведите нужную машину мышью. После анализа появится её видимый контур; скрытые части не восстанавливаются.')
        self.refresh_buttons();self.redraw()
    def clear_region(self):
        if self.busy:return
        self.region=None;self.region_start=None;self.scope.set(SCOPES[0]);self.scope_changed();self.status.set('Область снята. Нажмите «Анализировать», чтобы обработать весь кадр.')
    def region_position(self,event):
        if self.image is None or self.preview not in self.transforms:return None
        sx,sy,ox,oy=self.transforms[self.preview]
        return (max(0,min(self.image.width,(event.x-ox)/sx)),max(0,min(self.image.height,(event.y-oy)/sy)))
    def hit_vehicle(self,row,x,y):
        if row.get('segmentation'):
            from .segmentation import contains_point
            return contains_point(row['segmentation'],x,y)
        return row['box'][0]<=x<=row['box'][2] and row['box'][1]<=y<=row['box'][3]

    def start_region(self,event):
        if self.busy:return
        point=self.region_position(event)
        if point is None:return
        if self.scope.get()==SCOPES[0]:
            for item in reversed(self.preview.find_overlapping(event.x,event.y,event.x,event.y)):
                tag=next((t for t in self.preview.gettags(item) if t.startswith('car-') and t[4:].isdigit()),None)
                if tag:self.select_car(int(tag[4:]));return
            hits=[r for r in self.result['cars'] if self.hit_vehicle(r,*point)] if self.result else []
            if hits:
                row=min(hits,key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]));self.table.selection_set(str(row['id']));self.table.see(str(row['id']));self.redraw()
        else:self.region_start=point;self.region=None
    def drag_region(self,event):
        if self.busy or self.region_start is None:return
        point=self.region_position(event)
        if point is None:return
        x,y=self.region_start;x2,y2=point;self.region=[round(min(x,x2)),round(min(y,y2)),round(max(x,x2)),round(max(y,y2))];self.draw_region()
    def end_region(self,event):
        if self.region_start is None:return
        self.drag_region(event);self.region_start=None
        if self.region and min(self.region[2]-self.region[0],self.region[3]-self.region[1])<16:self.region=None;self.status.set('Выделите область не меньше 16 × 16 пикселей.')
        else:self.status.set('Область выбрана. Нажмите «Анализировать».')
        self.draw_region();self.refresh_buttons()
    def draw_region(self):
        self.preview.delete('region')
        if self.region is not None and self.scope.get()!=SCOPES[0] and self.preview in self.transforms:
            sx,sy,ox,oy=self.transforms[self.preview];x,y,x2,y2=self.region;self.preview.create_rectangle(ox+x*sx,oy+y*sy,ox+x2*sx,oy+y2*sy,outline='#ffb341',width=3,dash=(6,3),tags='region')
    def zoom_car(self):
        if self.busy or self.image is None:return
        selection=self.table.selection();row=next((r for r in self.result['cars'] if str(r['id'])==selection[0]),None) if selection and self.result else None
        box=self.region if self.scope.get()!=SCOPES[0] and self.region else row['box'] if row else None
        if box is None:return
        if self.zoom_window is not None and self.zoom_window.winfo_exists():self.zoom_window.destroy()
        win=tk.Toplevel(self.root);self.zoom_window=win;win.title('Выбранная машина');win.geometry('820x640');win.minsize(480,360)
        caption=(f"№ {row['id']} · {vehicle_name(row)} · {COLOUR_RU.get(row['colour'],row['colour'])}" if row and box==row['box'] else 'Выделенная область')
        ttk.Label(win,text=caption,font=('Segoe UI',15,'bold'),padding=12).pack(fill='x');canvas=tk.Canvas(win,bg=NAVY,highlightthickness=0);canvas.pack(fill='both',expand=True);crop=self.image.crop(box)
        if row and row.get('segmentation') and box==row['box']:
            from .segmentation import decode_mask,isolate
            crop=isolate(crop,decode_mask(row['segmentation']))
        canvas.bind('<Configure>',lambda e:self.draw(canvas,crop,enlarge=True));win.after_idle(lambda:self.draw(canvas,crop,enlarge=True));win.bind('<Destroy>',lambda e:self.transforms.pop(canvas,None) if e.widget==win else None)
        def use_region():
            if self.busy:return
            self.region=list(box);self.scope.set(SCOPES[2]);self.scope_changed();win.destroy();self.predict()
        ttk.Button(win,text='Распознать эту машину отдельно',command=use_region).pack(pady=10)
    def copy(self):
        if self.result is None:return
        text='; '.join(r['brand']+' '+r['colour'] for r in self.result['cars']) or 'No cars detected';self.root.clipboard_clear();self.root.clipboard_append(text);self.status.set('Скопированы пары «марка цвет» в формате задания. Модели и варианты доступны отдельной кнопкой.')
    def copy_details(self):
        if self.result is None:return
        text='\n'.join(detailed_answer(r) for r in self.result['cars']) or 'No cars detected';self.root.clipboard_clear();self.root.clipboard_append(text);self.status.set('Скопированы модели, варианты и уверенность.')
    def export(self):
        if self.result is None:return
        path=filedialog.askdirectory(title='Сохранить TXT, JSON и фото с рамками',initialdir=str(ROOT/'outputs'))
        if path:
            try:save_prediction(self.path,self.result,path);self.status.set('Файлы экспортированы: '+path)
            except Exception as e:messagebox.showerror('Экспорт',str(e))

    def open_label(self):
        if self.busy:return
        path=filedialog.askopenfilename(title='Фотография для разметки',filetypes=[('Фотографии','*.jpg *.jpeg *.png *.bmp *.webp')])
        if path:self.load_label_photo(path)
    def load_label_photo(self,path):
        try:
            image=read_image(path);digest=hashlib.sha256(str(image.size).encode()+image.tobytes()).hexdigest()[:20];directory=ROOT/'data/images';directory.mkdir(parents=True,exist_ok=True);self.label_file=directory/(digest+'.png')
            if not self.label_file.exists():image.save(self.label_file)
            self.label_image=image;self.box=None;self.start=None
            if not self.keep_group.get() or not self.group.get():self.group.set(digest)
            self.reload_labels();self.show('labels');self.refresh_buttons();self.redraw()
        except Exception as e:messagebox.showerror('Разметка',str(e))
    def all_labels(self):
        path=ROOT/'data/labels.jsonl';return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()] if path.exists() else []
    def write_labels(self,rows):
        path=ROOT/'data/labels.jsonl';path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8');temp.replace(path)
    def reload_labels(self):self.label_rows=[r for r in self.all_labels() if r['image']=='images/'+self.label_file.name];self.label_status.set(f'Сохранено машин: {len(self.label_rows)}. Наведите на номер, чтобы посмотреть марку и цвет. Можно обвести следующую машину.')
    def hover_label(self,event):
        for item in reversed(self.canvas.find_overlapping(event.x,event.y,event.x,event.y)):
            tag=next((t for t in self.canvas.gettags(item) if t.startswith('car-') and t[4:].isdigit()),None)
            if tag and 0<int(tag[4:])<=len(self.label_rows):
                row=self.label_rows[int(tag[4:])-1];self.label_status.set(f"Машина № {tag[4:]}: {vehicle_name(row)} · {COLOUR_RU.get(row['colour'],row['colour'])}. Сохранённая учебная метка.");return
    def position(self,event):
        if self.canvas not in self.transforms:return None
        sx,sy,ox,oy=self.transforms[self.canvas];return max(0,min(self.label_image.width,(event.x-ox)/sx)),max(0,min(self.label_image.height,(event.y-oy)/sy))
    def start_box(self,event):
        if not self.busy and self.label_image is not None:self.start=self.position(event);self.box=None;self.refresh_buttons()
    def drag_box(self,event):
        if self.busy or self.start is None:return
        end=self.position(event)
        if end is None:return
        x,y=self.start;x2,y2=end;self.box=[round(min(x,x2)),round(min(y,y2)),round(max(x,x2)),round(max(y,y2))];self.draw_selection()
    def end_box(self,event):
        self.drag_box(event);self.start=None
        if self.box and min(self.box[2]-self.box[0],self.box[3]-self.box[1])<16:self.box=None;self.canvas.delete('selection');self.status.set('Рамка слишком мала: минимум 16 × 16 пикселей исходного снимка.')
        self.refresh_buttons()
    def draw_selection(self):
        self.canvas.delete('selection')
        if self.box is not None and self.canvas in self.transforms:
            sx,sy,ox,oy=self.transforms[self.canvas];x,y,x2,y2=self.box;self.canvas.create_rectangle(ox+x*sx,oy+y*sy,ox+x2*sx,oy+y2*sy,outline='#ffb341',width=3,tags='selection')
    def save_label(self):
        if self.busy or self.box is None:return
        brand,colour,group,source=self.brand.get().strip(),self.colour.get(),self.group.get().strip(),self.source.get().strip()
        if not brand or not group or not source or colour not in COLOURS:messagebox.showerror('Метка','Заполните марку, цвет, группу и источник.');return
        row=dict(image='images/'+self.label_file.name,box=self.box.copy(),brand=brand,colour=colour,group=group,source=source)
        try:
            rows=self.all_labels()
            if any(r['image']==row['image'] and r['box']==row['box'] for r in rows):messagebox.showinfo('Метка уже есть','Такая рамка уже сохранена.');return
            rows.append(row);self.write_labels(rows);self.undo_rows.append(row);self.box=None;self.reload_labels();self.refresh_buttons();self.redraw();self.status.set('Метка сохранена: '+brand+' '+colour)
        except Exception as e:messagebox.showerror('Сохранение',str(e))
    def undo_label(self):
        if self.busy or not self.undo_rows:return
        try:
            rows=self.all_labels();row=self.undo_rows[-1]
            for i in range(len(rows)-1,-1,-1):
                if rows[i]==row:rows.pop(i);break
            self.write_labels(rows);self.undo_rows.pop()
            if self.label_image is not None:self.reload_labels()
            self.refresh_buttons();self.redraw();self.status.set('Последняя метка отменена; исходное фото сохранено.')
        except Exception as e:messagebox.showerror('Отмена',str(e))
    def train(self):
        if self.busy:return
        try:
            epochs,detector_epochs=self.epochs.get(),self.detector_epochs.get()
            if min(epochs,detector_epochs)<1:raise ValueError()
        except Exception:messagebox.showerror('Эпохи','Введите положительные целые числа для обеих сетей.');return
        self.command(['train','--labels',str(ROOT/'data/labels.jsonl'),'--output',str(ROOT/'models/own_training_candidate'),'--epochs',str(epochs),'--detector-epochs',str(detector_epochs)])
    def append_log(self,text):self.log.configure(state='normal');self.log.insert('end',text);self.log.see('end');self.log.configure(state='disabled')
    def command(self,args):
        if self.busy:return
        self.cancelled=False;self.set_busy(True,'Выполняется '+args[0]+'…');self.append_log('\n> '+args[0]+'\n');executable=Path(sys.executable)
        if not getattr(sys,'frozen',False) and executable.name.lower()=='pythonw.exe':executable=executable.with_name('python.exe')
        cmd=[str(executable)]+([] if getattr(sys,'frozen',False) else [str(ROOT/'launch.py')])+args
        def work():
            try:
                env=os.environ.copy();env['PYTHONIOENCODING']='utf-8';env['PYTHONUNBUFFERED']='1';proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=env);self.process=proc;self.events.put(('started',None))
                if self.cancelled or self.closed:proc.terminate()
                for line in proc.stdout:self.events.put(('log',line))
                self.events.put(('done',proc.wait()))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()
    def cancel(self):
        if self.process is not None and self.process.poll() is None:self.cancelled=True;self.process.terminate();self.status.set('Останавливаю процесс…')
    def poll(self):
        if self.closed:return
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='prediction':
                    self.result=data;self.table.delete(*self.table.get_children());self.show_evidence()
                    for row in data['cars']:self.table.insert('','end',iid=str(row['id']),values=(row['id'],vehicle_name(row) if row.get('make_first') or row['brand']!='unknown' or row.get('model_candidate','unknown')!='unknown' else 'Не уверен',('≈ ' if not row.get('colour_reliable',True) else '')+('Серый' if row['colour']=='grey' else COLOUR_RU.get(row['colour'],row['colour']))))
                    scope='Одна выбранная машина' if data.get('single_car_region') else 'Машины в области' if data.get('region') else 'Весь кадр'
                    self.summary.set(f"{scope} · машин: {len(data['cars'])}\nАнализ: {data['seconds']:.2f} с");self.set_busy(False,'Результаты сохранены в outputs.');self.redraw()
                    if data['cars']:
                        main=max(data['cars'],key=lambda r:r.get('mask_pixels',(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1])));self.table.selection_set(str(main['id']));self.table.see(str(main['id']));self.redraw()
                elif kind=='log':self.append_log(data)
                elif kind=='started':self.refresh_buttons()
                elif kind=='done':self.process=None;self.pipeline=None;self.set_busy(False,'Остановлено. Незавершённое обучение нужно повторить.' if self.cancelled else 'Готово.' if data==0 else 'Ошибка. Подробности в журнале.');self.append_log(f'\nКод завершения: {data}\n')
                elif kind=='error':self.process=None;self.set_busy(False,'Ошибка выполнения.');messagebox.showerror('CarVision',data)
        except queue.Empty:pass
        self.root.after(80,self.poll)
    def close(self):
        self.closed=True;self.cancelled=True
        if self.process is not None and self.process.poll() is None:self.process.terminate()
        self.root.destroy()

def run():
    root=tk.Tk();App(root);root.mainloop()
