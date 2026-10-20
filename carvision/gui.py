import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import ImageTk
from .core import ROOT, BRANDS, COLOURS, Pipeline, read_image, save_prediction

class App:
    def __init__(self,root):
        self.root=root; root.title('Tashkent CarVision | Offline Studio'); root.geometry('1200x820'); root.minsize(1000,700)
        self.events=queue.Queue(); self.pipeline=None; self.image=None; self.path=None; self.box=None; self.result=None; self.busy=False
        style=ttk.Style(); style.theme_use('clam')
        style.configure('TFrame',background='#eef2f6'); style.configure('TLabel',background='#eef2f6',font=('Segoe UI',10)); style.configure('TButton',font=('Segoe UI',10),padding=8)
        style.configure('Title.TLabel',font=('Segoe UI',22,'bold'),foreground='#132b45'); style.configure('Sub.TLabel',foreground='#536779')
        outer=ttk.Frame(root,padding=18); outer.pack(fill='both',expand=True)
        ttk.Label(outer,text='TASHKENT / CARVISION',style='Title.TLabel').pack(anchor='w')
        ttk.Label(outer,text='Локальный анализ фотографий • разметка • обучение • подготовка к экзамену',style='Sub.TLabel').pack(anchor='w',pady=(0,10))
        trained=(ROOT/'models/classifier.pt').exists()
        self.status=tk.StringVar(value='Классификатор обучен' if trained else 'Нужны ваши данные и обучение: марки пока выводятся как unknown')
        ttk.Label(outer,textvariable=self.status,foreground='#a65300' if not trained else '#087d6d').pack(anchor='w',pady=(0,10))
        tabs=ttk.Notebook(outer); tabs.pack(fill='both',expand=True)
        infer=ttk.Frame(tabs,padding=10); label=ttk.Frame(tabs,padding=10); train=ttk.Frame(tabs,padding=14)
        tabs.add(infer,text='  Распознавание  '); tabs.add(label,text='  Разметка данных  '); tabs.add(train,text='  Обучение и проверка  ')
        toolbar=ttk.Frame(infer); toolbar.pack(fill='x')
        ttk.Button(toolbar,text='Открыть фото',command=self.open_infer).pack(side='left')
        ttk.Button(toolbar,text='Распознать',command=self.predict).pack(side='left',padx=6)
        ttk.Button(toolbar,text='Копировать ответ',command=self.copy).pack(side='left')
        ttk.Button(toolbar,text='Папка результатов',command=lambda:self.open_folder(ROOT/'outputs')).pack(side='left',padx=6)
        self.det=tk.DoubleVar(value=.6); self.conf=tk.DoubleVar(value=.65)
        for text,var in [('Детектор',self.det),('Марка',self.conf)]:
            ttk.Label(toolbar,text=text).pack(side='left',padx=(10,2)); ttk.Spinbox(toolbar,from_=.1,to=.99,increment=.05,textvariable=var,width=5).pack(side='left')
        self.preview=tk.Canvas(infer,bg='#15263b',highlightthickness=0,height=390); self.preview.pack(fill='both',expand=True,pady=10)
        self.preview.create_text(500,170,text='Откройте фотографию с телефона',fill='#adc0d3',font=('Segoe UI',17))
        self.table=ttk.Treeview(infer,columns=('id','brand','colour','confidence'),show='headings',height=5)
        for name,title,width in [('id','№',50),('brand','Марка',260),('colour','Цвет',150),('confidence','Уверенность марки',180)]: self.table.heading(name,text=title); self.table.column(name,width=width)
        self.table.pack(fill='x')
        ttk.Label(infer,text='Результаты сохраняются автоматически: TXT, JSON и фото с рамками. Перезапуск фото на экзамене запрещён правилами задания.',style='Sub.TLabel').pack(anchor='w',pady=5)
        # Annotation coordinates are mapped back to the EXIF-corrected original.
        lt=ttk.Frame(label); lt.pack(fill='x')
        ttk.Button(lt,text='Добавить фото',command=self.open_label).pack(side='left')
        self.brand=tk.StringVar(value=BRANDS[0]); self.colour=tk.StringVar(value=COLOURS[0]); self.group=tk.StringVar(value=''); self.source=tk.StringVar(value='own')
        for title,var,values in [('Марка',self.brand,BRANDS),('Цвет',self.colour,COLOURS)]:
            ttk.Label(lt,text=title).pack(side='left',padx=5); ttk.Combobox(lt,textvariable=var,values=values,width=16,state='normal' if title=='Марка' else 'readonly').pack(side='left')
        ttk.Button(lt,text='Сохранить машину',command=self.save_label).pack(side='left',padx=6)
        meta=ttk.Frame(label); meta.pack(fill='x',pady=6)
        for title,var in [('Группа (одна машина/сессия)',self.group),('Источник (own/team/public + название)',self.source)]:
            ttk.Label(meta,text=title).pack(side='left',padx=4); ttk.Entry(meta,textvariable=var,width=23).pack(side='left')
        ttk.Label(label,text='Обведите каждую машину отдельно. Одна группа для повторных кадров той же машины/сессии; это предотвращает утечку в validation.').pack(anchor='w')
        self.canvas=tk.Canvas(label,bg='#15263b',highlightthickness=0); self.canvas.pack(fill='both',expand=True,pady=8)
        self.canvas.bind('<ButtonPress-1>',self.start_box); self.canvas.bind('<B1-Motion>',self.drag_box); self.canvas.bind('<ButtonRelease-1>',self.end_box)
        self.label_status=tk.StringVar(value='Фотографии копируются в data/images; исходные файлы сохраняются.'); ttk.Label(label,textvariable=self.label_status).pack(anchor='w')
        ttk.Label(train,text='Обучение на ваших фотографиях',style='Title.TLabel').pack(anchor='w')
        ttk.Label(train,text='Минимум две марки и независимые группы. Для полезной точности нужны сотни машин на марку.\nCOCO-детектор и ImageNet-веса уже входят в поставку. Интернет при распознавании не используется.').pack(anchor='w',pady=12)
        settings=ttk.Frame(train); settings.pack(fill='x')
        self.epochs=tk.IntVar(value=25)
        ttk.Label(settings,text='Эпохи').pack(side='left'); ttk.Spinbox(settings,from_=1,to=300,textvariable=self.epochs,width=6).pack(side='left',padx=8)
        ttk.Button(settings,text='Проверить данные',command=lambda:self.command(['audit','--labels',str(ROOT/'data/labels.jsonl')])).pack(side='left',padx=4)
        ttk.Button(settings,text='Начать обучение',command=self.train).pack(side='left',padx=4)
        ttk.Button(settings,text='Оценить модель',command=lambda:self.command(['evaluate','--labels',str(ROOT/'data/labels.jsonl'),'--models',str(ROOT/'models'),'--output',str(ROOT/'outputs/evaluation')])).pack(side='left',padx=4)
        self.log=tk.Text(train,bg='#15263b',fg='#d6e4ef',font=('Consolas',10),wrap='word'); self.log.pack(fill='both',expand=True,pady=15)
        self.log.insert('end','Здесь будут журнал обучения, метрики и ошибки.\nМарки без обучения не угадываются. Синтетические тесты не являются оценкой точности.\n')
        root.after(100,self.poll)

    def open_folder(self,path):
        import os
        path.mkdir(parents=True,exist_ok=True); os.startfile(path)

    def display(self,canvas,image):
        self.root.update_idletasks(); w=max(100,canvas.winfo_width()); h=max(100,canvas.winfo_height())
        im=image.copy(); im.thumbnail((w,h)); photo=ImageTk.PhotoImage(im)
        canvas.delete('all'); canvas.create_image(0,0,anchor='nw',image=photo); canvas.photo=photo
        return im.width/image.width,im.width,im.height

    def open_infer(self):
        if self.busy: return
        path=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp'),('All','*.*')])
        if not path:return
        try:
            self.image=read_image(path); self.path=path; self.result=None; self.display(self.preview,self.image)
            self.table.delete(*self.table.get_children()); self.status.set(Path(path).name+' — готово к анализу')
        except Exception as e: messagebox.showerror('Ошибка фото',str(e))

    def predict(self):
        if not self.path or self.busy:return
        try:
            d,b=self.det.get(),self.conf.get()
            if not 0<d<1 or not 0<b<1: raise ValueError('Пороги должны быть между 0 и 1')
        except Exception as e: messagebox.showerror('Пороги',str(e)); return
        self.busy=True; self.status.set('Анализ на процессоре…')
        source=self.path
        def work():
            try:
                if self.pipeline is None: self.pipeline=Pipeline()
                self.pipeline.detection=d; self.pipeline.brand_threshold=b
                result=self.pipeline.predict(source); annotated=save_prediction(source,result,ROOT/'outputs')
                self.events.put(('prediction',(result,annotated)))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()

    def copy(self):
        if self.result is None:return
        text='; '.join(r['brand']+' '+r['colour'] for r in self.result['cars']) or 'No cars detected'
        self.root.clipboard_clear(); self.root.clipboard_append(text)

    def open_label(self):
        path=filedialog.askopenfilename(filetypes=[('Images','*.jpg *.jpeg *.png *.bmp *.webp')])
        if not path:return
        try:
            self.label_image=read_image(path)
            import hashlib
            digest=hashlib.sha256(self.label_image.tobytes()).hexdigest()[:20]
            directory=ROOT/'data/images'; directory.mkdir(parents=True,exist_ok=True)
            self.label_file=directory/(digest+'.png')
            if not self.label_file.exists():self.label_image.save(self.label_file)
            self.scale,self.lw,self.lh=self.display(self.canvas,self.label_image); self.box=None
            if not self.group.get():self.group.set(digest)
            self.label_status.set('Обведите машину, проверьте марку и цвет, сохраните. '+self.label_file.name)
        except Exception as e:messagebox.showerror('Ошибка',str(e))

    def start_box(self,e):
        if not hasattr(self,'label_image'):return
        self.start=(max(0,min(e.x,self.lw)),max(0,min(e.y,self.lh))); self.canvas.delete('selection'); self.box=None
    def drag_box(self,e):
        if not hasattr(self,'start'):return
        self.canvas.delete('selection'); self.canvas.create_rectangle(*self.start,max(0,min(e.x,self.lw)),max(0,min(e.y,self.lh)),outline='#2dd4bf',width=3,tags='selection')
    def end_box(self,e):
        if not hasattr(self,'start'):return
        x,y=self.start; x2,y2=max(0,min(e.x,self.lw)),max(0,min(e.y,self.lh))
        self.box=[round(v/self.scale) for v in (min(x,x2),min(y,y2),max(x,x2),max(y,y2))]
        self.box[2]=min(self.box[2],self.label_image.width); self.box[3]=min(self.box[3],self.label_image.height)
    def save_label(self):
        if self.box is None or self.box[2]-self.box[0]<16 or self.box[3]-self.box[1]<16:messagebox.showerror('Рамка','Обведите машину размером хотя бы 16×16 пикселей.'); return
        if not self.brand.get().strip() or not self.group.get().strip() or not self.source.get().strip():messagebox.showerror('Метка','Заполните марку, группу и источник.'); return
        r=dict(image='images/'+self.label_file.name,box=self.box,brand=self.brand.get().strip(),colour=self.colour.get(),group=self.group.get().strip(),source=self.source.get().strip())
        with (ROOT/'data/labels.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        self.canvas.itemconfigure('selection',outline='#fbbf24'); self.label_status.set('Сохранено: '+r['brand']+' '+r['colour']+' — можно обвести следующую машину'); self.box=None

    def train(self):
        if self.busy:return
        try: epochs=self.epochs.get(); assert epochs>0
        except Exception:messagebox.showerror('Эпохи','Введите положительное целое число');return
        self.command(['train','--labels',str(ROOT/'data/labels.jsonl'),'--output',str(ROOT/'models'),'--epochs',str(epochs)])

    def command(self,args):
        if self.busy:return
        self.busy=True; self.status.set('Выполняется '+args[0]+'…')
        cmd=[sys.executable]+([] if getattr(sys,'frozen',False) else [str(ROOT/'launch.py')])+args
        def work():
            try:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                for line in proc.stdout:self.events.put(('log',line))
                code=proc.wait(); self.events.put(('done',code))
            except Exception as e:self.events.put(('error',str(e)))
        threading.Thread(target=work,daemon=True).start()

    def poll(self):
        try:
            while True:
                kind,data=self.events.get_nowait()
                if kind=='prediction':
                    self.result,annotated=data; self.busy=False; self.table.delete(*self.table.get_children())
                    for r in self.result['cars']:self.table.insert('', 'end',values=(r['id'],r['brand'],r['colour'],f"{r['brand_confidence']:.0%}"))
                    self.display(self.preview,read_image(annotated))
                    self.status.set(f"Машин: {len(self.result['cars'])} • {self.result['seconds']:.1f} с"+(' • ВНИМАНИЕ: марки unknown, модель не обучена' if not self.result['classifier_trained'] else ''))
                elif kind=='log':self.log.insert('end',data); self.log.see('end')
                elif kind=='done':self.busy=False; self.pipeline=None; self.status.set('Готово' if data==0 else 'Операция завершилась с ошибкой — см. журнал')
                elif kind=='error':self.busy=False; self.status.set('Ошибка'); messagebox.showerror('CarVision',data)
        except queue.Empty:pass
        self.root.after(100,self.poll)

def run():
    root=tk.Tk(); App(root); root.mainloop()
