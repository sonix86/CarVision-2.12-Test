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
from PIL import ImageTk,Image
from .core import ROOT,BRANDS,COLOURS,Pipeline,read_image,save_prediction
from .overlays import marker_positions

BG='#f3f5f9';INK='#182a42';MUTED='#6e7d91';TEAL='#087f8c';NAVY='#14243a'
MODES={'Баланс · фрагменты':'balanced','Быстрый · весь кадр':'fast','Детальный · фрагменты и отражение':'detail'}
SCOPES=['Весь кадр','Машины в области','Одна машина в области']
COLOUR_RU=dict(zip(COLOURS,['Белый','Чёрный','Серый / серебро','Красный','Синий','Зелёный','Коричневый','Жёлтый']))

class App:
    def __init__(self,root):
        self.root=root;root.title('CarVision · София Рахматулаева');root.geometry('1320x860');root.minsize(1120,760)
        self.events=queue.Queue();self.pipeline=None;self.image=None;self.path=None;self.result=None;self.busy=False;self.process=None;self.closed=False;self.cancelled=False
        self.label_image=None;self.box=None;self.start=None;self.label_rows=[];self.undo_rows=[];self.transforms={};self.resize_job=None;self.buttons={}
        self.region=None;self.region_start=None;self.zoom_window=None
        style=ttk.Style();style.theme_use('clam');style.configure('.',font=('Segoe UI',10))
        style.configure('TFrame',background=BG);style.configure('TLabel',background=BG,foreground=INK)
        style.configure('Title.TLabel',font=('Segoe UI',24,'bold'));style.configure('Muted.TLabel',foreground=MUTED)
        style.configure('TButton',padding=(12,9),borderwidth=0,background='#e4eaf2',foreground=INK)
        style.configure('Primary.TButton',background=TEAL,foreground='white');style.map('Primary.TButton',background=[('active','#066774'),('disabled','#bdd3d6')])
        style.configure('Treeview',rowheight=32,font=('Segoe UI',10),background='white',fieldbackground='white');style.configure('Treeview.Heading',background='#e4eaf2',font=('Segoe UI',9,'bold'))
        style.map('Treeview',background=[('selected','#cbece6')],foreground=[('selected',INK)])
        rail=tk.Frame(root,bg=NAVY,width=210);rail.pack(side='left',fill='y');rail.pack_propagate(False)
        tk.Label(rail,text='CV /',font=('Segoe UI',30,'bold'),fg='#45d4c2',bg=NAVY).pack(anchor='w',padx=23,pady=(25,6))
        tk.Label(rail,text='TASHKENT\nCARVISION',font=('Segoe UI',14,'bold'),fg='white',bg=NAVY,justify='left').pack(anchor='w',padx=23)
        tk.Label(rail,text='OFFLINE STUDIO  /  1.3',font=('Segoe UI',8),fg='#8fa3bb',bg=NAVY).pack(anchor='w',padx=23,pady=(10,30))
        self.nav={};self.views={}
        for key,title in [('infer','01   Анализ фото'),('labels','02   Учебная разметка'),('training','03   Обучение'),('help','04   О проекте')]:
            b=tk.Button(rail,text=title,command=lambda k=key:self.show(k),bg=NAVY,fg='#bbcadb',activebackground='#26445e',activeforeground='white',font=('Segoe UI',11),anchor='w',relief='flat',padx=18,pady=14,cursor='hand2');b.pack(fill='x',padx=10,pady=3);self.nav[key]=b
        tk.Label(rail,text='Владелец проекта\nСофия Рахматулаева',font=('Segoe UI',10),fg='#c4d3e4',bg=NAVY,justify='left').pack(side='bottom',anchor='w',padx=20,pady=22)
        main=ttk.Frame(root,padding=(24,18));main.pack(side='left',fill='both',expand=True)
        self.title=tk.StringVar();ttk.Label(main,textvariable=self.title,style='Title.TLabel').pack(anchor='w')
        ttk.Label(main,text='Марка и цвет — автоматически · локально · без API',style='Muted.TLabel').pack(anchor='w',pady=(5,15))
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

    def build_infer(self):
        view=self.views['infer'];bar=ttk.Frame(view);bar.pack(fill='x')
        self.button(bar,'open','Открыть фото',self.open_infer,True,side='left');self.button(bar,'samples','Примеры Ташкента',lambda:self.open_infer(ROOT/'samples/images'),side='left',padx=7);self.button(bar,'pdf','PDF с примерами',self.open_pdf,side='left')
        area=ttk.Frame(view);area.pack(fill='x',pady=(10,0));self.scope=tk.StringVar(value=SCOPES[0]);ttk.Label(area,text='Что распознать:').pack(side='left',padx=(0,8))
        choice=ttk.Combobox(area,textvariable=self.scope,values=SCOPES,state='readonly',width=26);choice.pack(side='left');choice.bind('<<ComboboxSelected>>',lambda e:self.scope_changed());self.scope_choice=choice
        self.button(area,'clear_region','Весь кадр',self.clear_region,side='left',padx=7);self.button(area,'zoom','Увеличить машину',self.zoom_car,side='left')
        self.region_hint=tk.StringVar(value='Нажмите на рамку или строку результата, чтобы выбрать машину.');ttk.Label(view,textvariable=self.region_hint,style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(3,0))
        settings=ttk.Frame(view);settings.pack(fill='x',pady=12);self.mode=tk.StringVar(value=next(iter(MODES)))
        self.mode_choice=ttk.Combobox(settings,textvariable=self.mode,values=list(MODES),state='readonly',width=23);self.mode_choice.pack(side='left');self.mode.set('Быстрый · весь кадр');self.det=tk.DoubleVar(value=.5);self.conf=tk.DoubleVar(value=.5)
        for text,var in [('Порог рамки',self.det),('Порог марки',self.conf)]:ttk.Label(settings,text=text,style='Muted.TLabel').pack(side='left',padx=(10,4));ttk.Spinbox(settings,from_=.1,to=.99,increment=.05,textvariable=var,width=5).pack(side='left')
        self.button(settings,'predict','Анализировать',self.predict,True,side='right')
        body=ttk.Frame(view);body.pack(fill='both',expand=True);self.preview=tk.Canvas(body,bg='#e4eaf2',highlightthickness=0);self.preview.pack(side='left',fill='both',expand=True)
        right=ttk.Frame(body,padding=(14,0,0,0),width=300);right.pack(side='right',fill='y');right.pack_propagate(False)
        ttk.Label(right,text='Результаты',font=('Segoe UI',13,'bold')).pack(anchor='w');self.summary=tk.StringVar(value='Фотография ещё не выбрана');ttk.Label(right,textvariable=self.summary,style='Muted.TLabel',wraplength=275).pack(anchor='w',pady=(5,10))
        tablebox=ttk.Frame(right);tablebox.pack(fill='x');self.table=ttk.Treeview(tablebox,columns=('id','brand','colour'),show='headings',height=3,selectmode='browse')
        for name,title,width in [('id','№',30),('brand','Марка',130),('colour','Цвет',90)]:self.table.heading(name,text=title);self.table.column(name,width=width,minwidth=25,stretch=name!='id')
        self.table.pack(side='left',fill='x',expand=True);scroll=ttk.Scrollbar(tablebox,command=self.table.yview);scroll.pack(side='right',fill='y');self.table.configure(yscrollcommand=scroll.set);self.table.bind('<<TreeviewSelect>>',lambda e:self.redraw())
        self.details=tk.StringVar(value='Выберите строку, чтобы выделить машину на фото.');ttk.Label(right,textvariable=self.details,style='Muted.TLabel',font=('Segoe UI',9),wraplength=275).pack(anchor='w',pady=6)
        for key,text,fn in [('copy','Копировать ответ',self.copy),('export','Экспортировать файлы',self.export),('output','Открыть результаты',lambda:self.open_folder(ROOT/'outputs'))]:self.button(right,key,text,fn,fill='x',pady=(0,4)).configure(padding=(8,6))
        self.model_status=tk.StringVar();ttk.Label(view,textvariable=self.model_status,foreground='#a66718',wraplength=850).pack(before=body,anchor='w',pady=(0,8))
        ttk.Label(view,text='Ctrl+O — открыть фото · Ctrl+Enter — анализировать. Примеры не содержат ответов для модели.',style='Muted.TLabel').pack(anchor='w',pady=(9,0));self.preview.bind('<Configure>',lambda e:self.queue_redraw())
        self.preview.bind('<ButtonPress-1>',self.start_region);self.preview.bind('<B1-Motion>',self.drag_region);self.preview.bind('<ButtonRelease-1>',self.end_region);self.preview.bind('<Double-Button-1>',lambda e:self.zoom_car());self.table.bind('<Double-Button-1>',lambda e:self.zoom_car())

    def build_labels(self):
        view=self.views['labels'];ttk.Label(view,text='Подготовка учебных ответов. В «Анализе фото» программа определяет марку и цвет автоматически.',style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(0,10));bar=ttk.Frame(view);bar.pack(fill='x')
        self.button(bar,'label_open','Добавить своё фото',self.open_label,True,side='left');self.button(bar,'label_save','Сохранить машину',self.save_label,side='left',padx=7);self.button(bar,'undo','Отменить последнюю',self.undo_label,side='left')
        fields=ttk.Frame(view);fields.pack(fill='x',pady=12);self.brand=tk.StringVar(value=BRANDS[0]);self.colour=tk.StringVar(value=COLOURS[0]);self.group=tk.StringVar();self.source=tk.StringVar(value='own');self.keep_group=tk.BooleanVar()
        for title,var,values in [('Марка',self.brand,BRANDS),('Цвет',self.colour,COLOURS)]:ttk.Label(fields,text=title).pack(side='left',padx=(0,6));ttk.Combobox(fields,textvariable=var,values=values,state='normal' if title=='Марка' else 'readonly',width=18).pack(side='left',padx=(0,18))
        meta=ttk.Frame(view);meta.pack(fill='x',pady=(0,10))
        for title,var in [('Группа / сессия',self.group),('Источник',self.source)]:ttk.Label(meta,text=title,style='Muted.TLabel').pack(side='left',padx=(0,6));ttk.Entry(meta,textvariable=var,width=21).pack(side='left',padx=(0,14))
        ttk.Checkbutton(meta,text='Одна группа для серии',variable=self.keep_group).pack(side='left')
        self.canvas=tk.Canvas(view,bg='#e4eaf2',highlightthickness=0);self.canvas.pack(fill='both',expand=True)
        for event,fn in [('<ButtonPress-1>',self.start_box),('<B1-Motion>',self.drag_box),('<ButtonRelease-1>',self.end_box)]:self.canvas.bind(event,fn)
        self.canvas.bind('<Configure>',lambda e:self.queue_redraw());self.label_status=tk.StringVar(value='Обведите каждую машину. Зелёные рамки сохранены, жёлтая рамка — новая.');ttk.Label(view,textvariable=self.label_status,style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(10,0))

    def build_training(self):
        view=self.views['training'];ttk.Label(view,text='Разметка  →  Проверка данных  →  Обучение двух сетей  →  Проверка новых фото',font=('Segoe UI',12,'bold'),wraplength=850).pack(anchor='w',pady=(0,12))
        ttk.Label(view,text='Нужны минимум две марки и независимые группы. Используется локальная ImageNet/COCO-основа; новые головы обучаются на вашей разметке. Предыдущие модели сохраняются в models/backups. Нужен большой и разнообразный набор.',style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=(0,15))
        fields=ttk.Frame(view);fields.pack(fill='x');self.epochs=tk.IntVar(value=25);self.detector_epochs=tk.IntVar(value=25)
        for title,var in [('Эпохи классификатора',self.epochs),('Эпохи детектора',self.detector_epochs)]:ttk.Label(fields,text=title).pack(side='left',padx=(0,7));ttk.Spinbox(fields,from_=1,to=1000,textvariable=var,width=6).pack(side='left',padx=(0,20))
        bar=ttk.Frame(view);bar.pack(fill='x',pady=12)
        self.button(bar,'audit','Проверить данные',lambda:self.command(['audit','--labels',str(ROOT/'data/labels.jsonl')]),side='left');self.button(bar,'train','Обучить обе сети',self.train,True,side='left',padx=7);self.button(bar,'evaluate','Проверка качества',self.evaluate,side='left');self.button(bar,'cancel','Остановить',self.cancel,side='right')
        self.log=tk.Text(view,bg=NAVY,fg='#d6e4ef',font=('Consolas',10),relief='flat',wrap='word',padx=14,pady=14,state='disabled');self.log.pack(fill='both',expand=True);self.append_log('Обучение локально, без API. Общие веса уже включены в пакет.\nГотовые классификаторы марок и цветов не используются.\nОценка на вырезках не заменяет проверку полного конвейера на новых фото.\n')

    def build_help(self):
        view=self.views['help'];ttk.Label(view,text='София Рахматулаева',font=('Segoe UI',15,'bold')).pack(anchor='w',pady=(0,15))
        text='Откройте новую фотографию во вкладке «Анализ фото» и нажмите «Анализировать». Программа находит машины, затем определяет марку и цвет каждой вырезки. Правильную марку при распознавании вводить не нужно. Точная модель автомобиля не требуется.\n\nПоставляемый прототип обучен на открытых фотографиях. Поддерживаемые марки и измерения качества перечислены в инструкции. Неуверенная марка отображается как unknown; неизвестная классу машина тоже может быть ошибочно отнесена к известной марке. Цвет определяется собственным алгоритмом, без готового классификатора.\n\nДля задания нужны также ваши фотографии Ташкента: добавляйте их в «Учебной разметке», сохраняйте марки, цвета и группы съёмки. Это отдельная подготовка данных для обучения.\n\nДля переноса скопируйте всю папку CarVision, включая models и _internal.'
        ttk.Label(view,text=text,wraplength=850,justify='left').pack(anchor='w',pady=(0,18))
        for key,text,fn in [('readme','Полная инструкция',lambda:self.open_document(ROOT/'README.md')),('checklist','Список требований',lambda:self.open_document(ROOT/'CHECKLIST.md')),('help_pdf','PDF с примерами',self.open_pdf),('folder','Папка проекта',lambda:self.open_folder(ROOT))]:self.button(view,key,text,fn,anchor='w',pady=4)
        ttk.Label(view,text='Публичные примеры в samples не являются собственным обучающим набором Софии. Авторы, ссылки и лицензии сохранены. GitHub будет подключён после передачи ссылки владельцем.',style='Muted.TLabel',wraplength=850).pack(anchor='w',pady=20)

    def refresh_buttons(self):
        detector=(ROOT/'models/detector.pt').exists();classifier=(ROOT/'models/classifier.pt').exists()
        self.scope_choice.configure(state='disabled' if self.busy else 'readonly');self.mode_choice.configure(state='disabled' if self.busy else 'readonly')
        for key,b in self.buttons.items():
            enabled=not self.busy
            if key=='predict':enabled=enabled and self.path is not None and detector
            if key in ('copy','export'):enabled=enabled and self.result is not None
            if key=='zoom':enabled=enabled and (self.region is not None or bool(self.table.selection()))
            if key=='clear_region':enabled=enabled and (self.region is not None or self.scope.get()!=SCOPES[0])
            if key=='label_save':enabled=enabled and self.box is not None
            if key=='undo':enabled=enabled and bool(self.undo_rows)
            if key=='cancel':enabled=self.busy and self.process is not None
            b.configure(state='normal' if enabled else 'disabled')
        self.model_status.set('Готово. Выберите весь кадр или обведите нужную машину. Марка и цвет определяются автоматически.' if detector and classifier else 'Обучение не завершено: '+('нужны детектор и классификатор.' if not detector and not classifier else 'нужен детектор.' if not detector else 'нужен классификатор.'))
    def evaluate(self):
        if (ROOT/'models/evaluation/public_test.json').exists() and not self.all_labels():
            self.open_document(ROOT/'PROTOTYPE_VALIDATION.md')
        else:self.command(['evaluate','--labels',str(ROOT/'data/labels.jsonl'),'--models',str(ROOT/'models'),'--output',str(ROOT/'models/evaluation')])
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
        self.path=str(path);self.image=image;self.result=None;self.region=None;self.region_start=None;self.table.delete(*self.table.get_children());self.details.set('Выберите строку результата, чтобы выделить машину.');self.scope.set(SCOPES[0]);self.scope_changed()
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
        ox=(w-im.width)//2;oy=(h-im.height)//2;photo=ImageTk.PhotoImage(im);canvas.photo=photo;canvas.create_image(ox,oy,image=photo,anchor='nw')
        sx,sy=im.width/image.width,im.height/image.height;self.transforms[canvas]=(sx,sy,ox,oy)
        screen_boxes=[]
        for i,row in enumerate(rows or [],1):
            x,y,x2,y2=row['box'];coords=(ox+x*sx,oy+y*sy,ox+x2*sx,oy+y2*sy);ident=row.get('id',i)
            canvas.create_rectangle(*coords,outline='#ffb341' if ident==selected else '#24c9ad',width=3 if ident==selected else 2)
            screen_boxes.append(coords)
        for i,(row,rect) in enumerate(zip(rows or [],marker_positions(screen_boxes,w,h)),1):
            ident=row.get('id',i);x,y,x2,y2=rect;box=screen_boxes[i-1];canvas.create_line(box[0],box[1],(x+x2)/2,(y+y2)/2,fill='#24c9ad');canvas.create_rectangle(*rect,fill=TEAL if ident==selected else NAVY,outline='#ffb341' if ident==selected else '#24c9ad');canvas.create_text((x+x2)/2,(y+y2)/2,text=str(ident),fill='white',font=('Segoe UI',9,'bold'),tags='car-marker')
            if canvas==self.preview:
                tag=f'car-{ident}';canvas.addtag_withtag(tag,canvas.find_all()[-1]);canvas.addtag_withtag(tag,canvas.find_all()[-2]);canvas.tag_bind(tag,'<Button-1>',lambda event,k=ident:self.select_car(k))

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
                row=next(r for r in self.result['cars'] if r['id']==selected);self.details.set(f"{row['brand'] if row['brand']!='unknown' else 'Марка не определена'} · {COLOUR_RU.get(row['colour'],row['colour'])}\nУверенность марки: {row['brand_confidence']:.0%}\nУверенность цвета: {row.get('colour_confidence',0):.0%}\nДвойной щелчок — крупный просмотр.")
        elif self.current=='labels':self.draw(self.canvas,self.label_image,self.label_rows);self.draw_selection()
    def predict(self):
        if self.busy or not self.path:return
        try:
            d,b=self.det.get(),self.conf.get();mode=MODES[self.mode.get()]
            if not 0<d<1 or not 0<b<1:raise ValueError('Пороги должны быть между 0 и 1.')
        except Exception as e:messagebox.showerror('Настройки',str(e));return
        region=self.region.copy() if self.region is not None and self.scope.get()!=SCOPES[0] else None;single=self.scope.get()==SCOPES[2]
        if self.scope.get()!=SCOPES[0] and region is None:messagebox.showinfo('Выберите область','Обведите мышью машину или участок фотографии.');return
        source=self.path;self.set_busy(True,'Анализ выделенной области…' if region else 'Анализ фотографии…')
        def work():
            try:
                if self.pipeline is None:self.pipeline=Pipeline()
                self.pipeline.detection=d;self.pipeline.brand_threshold=b;self.pipeline.mode=mode;result=self.pipeline.predict(source,region=region,single=single);save_prediction(source,result,ROOT/'outputs');self.events.put(('prediction',result))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()
    def scope_changed(self):
        self.preview.configure(cursor='arrow' if self.scope.get()==SCOPES[0] else 'crosshair')
        self.region_hint.set('Нажмите на рамку или строку результата, чтобы выбрать машину.' if self.scope.get()==SCOPES[0] else 'Обведите нужный участок мышью. Марку и цвет программа выберет сама.')
        self.refresh_buttons();self.redraw()
    def clear_region(self):
        if self.busy:return
        self.region=None;self.region_start=None;self.scope.set(SCOPES[0]);self.scope_changed();self.status.set('Область снята. Нажмите «Анализировать», чтобы обработать весь кадр.')
    def region_position(self,event):
        if self.image is None or self.preview not in self.transforms:return None
        sx,sy,ox,oy=self.transforms[self.preview]
        return (max(0,min(self.image.width,(event.x-ox)/sx)),max(0,min(self.image.height,(event.y-oy)/sy)))
    def start_region(self,event):
        if self.busy:return
        point=self.region_position(event)
        if point is None:return
        if self.scope.get()==SCOPES[0]:
            hits=[r for r in self.result['cars'] if r['box'][0]<=point[0]<=r['box'][2] and r['box'][1]<=point[1]<=r['box'][3]] if self.result else []
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
        caption=(f"№ {row['id']} · {row['brand']} · {COLOUR_RU.get(row['colour'],row['colour'])}" if row and box==row['box'] else 'Выделенная область')
        ttk.Label(win,text=caption,font=('Segoe UI',15,'bold'),padding=12).pack(fill='x');canvas=tk.Canvas(win,bg=NAVY,highlightthickness=0);canvas.pack(fill='both',expand=True);crop=self.image.crop(box)
        canvas.bind('<Configure>',lambda e:self.draw(canvas,crop,enlarge=True));win.after_idle(lambda:self.draw(canvas,crop,enlarge=True));win.bind('<Destroy>',lambda e:self.transforms.pop(canvas,None) if e.widget==win else None)
        def use_region():
            if self.busy:return
            self.region=list(box);self.scope.set(SCOPES[2]);self.scope_changed();win.destroy();self.predict()
        ttk.Button(win,text='Распознать эту машину отдельно',command=use_region).pack(pady=10)
    def copy(self):
        if self.result is None:return
        text='; '.join(r['brand']+' '+r['colour'] for r in self.result['cars']) or 'No cars detected';self.root.clipboard_clear();self.root.clipboard_append(text);self.status.set('Ответ скопирован.')
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
    def reload_labels(self):self.label_rows=[r for r in self.all_labels() if r['image']=='images/'+self.label_file.name];self.label_status.set(f'Сохранено машин на фото: {len(self.label_rows)}. Зелёные рамки уже сохранены; можно обвести следующую машину.')
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
        self.command(['train','--labels',str(ROOT/'data/labels.jsonl'),'--output',str(ROOT/'models'),'--epochs',str(epochs),'--detector-epochs',str(detector_epochs)])
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
                    self.result=data;self.table.delete(*self.table.get_children())
                    for row in data['cars']:self.table.insert('','end',iid=str(row['id']),values=(row['id'],'Не уверен' if row['brand']=='unknown' else row['brand'],'Серый' if row['colour']=='grey' else COLOUR_RU.get(row['colour'],row['colour'])))
                    scope='Одна выбранная машина' if data.get('single_car_region') else 'Машины в области' if data.get('region') else 'Весь кадр'
                    self.summary.set(f"{scope} · машин: {len(data['cars'])}\nАнализ: {data['seconds']:.2f} с");self.set_busy(False,'Результаты сохранены в outputs.');self.redraw()
                    if data['cars']:
                        main=max(data['cars'],key=lambda r:(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1]));self.table.selection_set(str(main['id']));self.table.see(str(main['id']));self.redraw()
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
