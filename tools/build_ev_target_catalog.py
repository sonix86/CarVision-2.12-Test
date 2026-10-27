"""Track requested coverage without treating catalogue names as trained classes.

Names originate from the user's requested scope, not verified specifications.
Prices, market prevalence, and propulsion claims are deliberately not inferred.
"""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REQUESTED='''
1|Wuling|Hongguang Mini EV;Air EV;Bingo;Bingo Plus;Nano EV
1|Changan|Lumin;BenBen E-Star
1|Dongfeng|EX1;Nano Box;Fengon Mini EV
1|Nammi|01
1|Chery|eQ1;QQ Ice Cream;Wujie Pro
1|Geely|Panda Mini;Panda Knight
1|Geometry|E;EX3
1|BYD|e1;S2;Seagull
1|Baojun|Kiwi EV;E100;E200;Yep;Yep Plus
1|Leapmotor|T03
1|Neta|V;Aya
1|JAC|iEV6E;iEV7S
1|BAIC|EC3;EC5;Lite
1|Bestune|Pony
1|Ora|Black Cat;White Cat
1|Pocco|Meimei;Duoduo
1|Letin|Mengo
1|Hanteng|X5 EV
1|Zotye|E200
2|BYD|Seagull;e2;Dolphin;Yuan Up;Yuan Pro;Qin Plus;Chazor;D1;e3;Qin EV
2|GAC Aion|S;S Plus;S Max;Y;Y Plus;Y Younger;UT
2|Changan|Eado EV;Qiyuan A05;Qiyuan Q05;CS15 E-Pro;CS55 E-Rock
2|Geometry|A;A Pro;C;M6;G6
2|Geely|Emgrand EV;GSE;Galaxy L6;Starwish
2|BAIC|EU5;EU5 Plus;EU7;EX360
2|Aeolus|E70;Sky EV01
2|Dongfeng|Nammi 06;Box
2|Neta|U;U-II;X
2|Volkswagen|ID.3;e-Lavida;e-Bora;e-Golf
2|Chevrolet|Menlo;Bolt EV
2|Buick|Velite 6;Velite 7
2|MG|MG4 EV;ZS EV;MG5 EV
2|JAC|iC5;Sehol E40X;Yiwei 3
2|Wuling|Starlight;Starlight S
2|Ora|Good Cat;Good Cat GT
2|Nissan|Sylphy Zero Emission;Leaf
2|Everus|VE-1
2|Ciimo|M-NV;X-NV
3|BYD|Song Plus;Yuan Plus;Song Pro;Seal 06 DM-i;Seal 06 GT;Qin L;Seal;Seal DM-i;Song L DM-i
3|Deepal|S07;SL03;L07;S05
3|Changan|Nevo A07;UNI-K iDD;UNI-Z iDD;UNI-V iDD;UNI-E
3|Leapmotor|C11;C10;C16;C01;B10
3|Volkswagen|ID.4 Crozz;ID.4 X;ID.6 Crozz;ID.6 X;ID.7 Vizzion;ID.UNYX
3|Geely|Galaxy E5;Galaxy L7;Galaxy E8
3|Honda|e:NS1;e:NP1;e:NS2;e:NP2
3|Toyota|bZ4X;bZ3;bZ3X
3|Dongfeng|eπ 007;eπ 008
3|Forthing|Friday;Xinghai S7
3|iCAR|03;V23
3|Chery|eQ7;Fulwin A8;Fulwin T9;Fulwin T10
3|Jetour|Shanhai T2;Shanhai L9
3|Hongqi|E-QM5;EHS3
3|XPeng|Mona M03;P5;G3;G3i;G6
3|GAC Aion|V;V Plus;LX;i60
3|Arcfox|Kaola S;Alpha T5;Alpha S5
3|Haval|H6 PHEV;Xiaolong Max;Raptor
3|Wey|Mocha DHT-PHEV
3|Skywell|ET5;HT-i
3|Weltmeister|EX5;W6
3|Neta|S;GT
3|Rising Auto|F7;R7
3|Roewe|D7;Marvel R
3|Mazda|EZ-6
4|Zeekr|001;007;7X;X;Mix
4|Li Auto|L6;L7;L8;L9;One
4|BYD|Han;Tang;Song L EV;Sealion 07 EV;Xia;Atto 8;Ti7
4|FangChengBao|Bao 5;Bao 3
4|Denza|N7;N8
4|Xiaomi|SU7;YU7
4|Avatr|11;12;07
4|Aito|M5;M7
4|Voyah|Free;Passion;Courage
4|Exlantix|ES;ET
4|Luxeed|S7;R7
4|IM Motors|L6;L7;LS6;LS7
4|Nio|ET5;ET5T;ET7;ES6;EC6;ES7;EC7;ES8
4|Onvo|L60
4|XPeng|P7;P7i;G9;X9
4|Deepal|G318
4|Tank|400 Hi4-T;500 Hi4-T;700 Hi4-T
4|Jiyue|01;07
4|Hyptec|GT;HT
4|Lynk & Co|Z10;Z20;08 EM-P;09 EM-P
4|Smart|#1;#3;#5
4|Polestar|2;4
4|MG|Cyberster
4|Tesla|Model 3;Model Y
4|BMW|i3 Sedan;i4;iX3
4|Kia|EV5;EV6;EV9
4|Hyundai|Ioniq 5;Ioniq 6
4|Audi|Q4 e-tron;Q5 e-tron
5|YangWang|U8;U9;U7
5|Zeekr|009;009 Grand;001 FR;8X
5|Denza|Z9 GT;Z9;D9;N9
5|FangChengBao|Bao 8
5|Li Auto|Mega
5|Aito|M9
5|Stelato|S9
5|Maextro|S800
5|M-Hero|917
5|Voyah|Dream
5|Hongqi|E-HS9
5|Lotus|Eletre;Emeya
5|HiPhi|Z;X;Y
5|Xiaomi|SU7 Ultra
5|Hyptec|SSR
5|Nio|ET9
5|Polestar|3
5|LEVC|L380
5|Tesla|Cybertruck;Model S;Model X
5|Mercedes-Benz|G 580 EQ;EQS;EQS SUV;EQE;EQE SUV;EQC;EQA;EQB;Maybach EQS SUV
5|Porsche|Taycan;Macan Electric
5|BMW|i7;iX;i5
5|Audi|e-tron GT;RS e-tron GT;Q8 e-tron;e-tron S
5|GMC|Hummer EV
5|Rivian|R1T;R1S
5|Cadillac|Lyriq;Escalade IQ
5|Lucid|Air
5|Rolls-Royce|Spectre
'''
ALIASES={
 'Wuling Hongguang Mini EV':['Hongguang Mini EV Standard','Macaron','Gameboy Edition'],
 'Changan Lumin':['Lumin Corn'], 'Nammi 01':['Nammi Box'],
 'Dongfeng EX1':['EX1 Pro','Renault City K-ZE'],
 'Chery eQ1':['Little Ant','Муравей'], 'Chery Wujie Pro':['Unbounded Pro'],
 'Geometry E':['Firefly'], 'Geometry EX3':['Kungfu Cow'],
 'Baojun Kiwi EV':['E300'], 'JAC iEV6E':['Sehol E10X'],
 'Bestune Pony':['Xiaoma'], 'Ora Black Cat':['R1'], 'Ora White Cat':['R2'],
 'BYD Chazor':['Destroyer 05'], 'BYD Yuan Plus':['Atto 3'],
 'Geely Galaxy E5':['Geely EX5'], 'Geely Starwish':['Xingyuan'],
 'Deepal S07':['S7','Shenlan S7'], 'Changan Nevo A07':['Qiyuan A07'],
 'Changan Qiyuan A05':['Nevo A05'], 'Changan Qiyuan Q05':['Nevo Q05'],
 'MG MG4 EV':['Mulan'], 'JAC iC5':['Sehol E50A'], 'Ora Good Cat':['Haomao'],
 'BYD Seal DM-i':['Chaser 07'], 'Forthing Friday':['Fengxing Leiting'],
 'Chery eQ7':['Shuxiangjia'], 'Haval Raptor':['Menglong'],
 'FangChengBao Bao 5':['Leopard 5'], 'FangChengBao Bao 3':['Tai 3'],
 'FangChengBao Bao 8':['Leopard 8'], 'Voyah Passion':['Zhuiguang'],
 'Voyah Courage':['Zhiyin'], 'Voyah Dream':['Dreamer'],
 'Exlantix ES':['Sterra ES'], 'Exlantix ET':['Sterra ET'],
 'BYD Song Plus':['EV','DM-i','Champion','UzPride'],
 'BYD Qin Plus':['EV','DM-i'], 'BYD Han':['EV','DM-i'],
 'BYD Tang':['EV','DM-i'], 'Nissan Leaf':['ZE0','ZE1'],
 'Li Auto L7':['LiXiang L7'], 'Xiaomi SU7':['Standard','Pro','Max'],
 'Tesla Model 3':['Highland'], 'Tesla Model S':['Plaid'], 'Tesla Model X':['Plaid'],
 'Porsche Taycan':['4S','Turbo','Turbo S','Cross Turismo'],
}
data=json.loads((ROOT/'models/ev_patch_v22/dataset.json').read_text(encoding='utf-8'))
supported={' '.join(p) for p in data['pairs']}
catalog={}
for line in REQUESTED.strip().splitlines():
    tier,brand,models=line.split('|')
    for model in models.split(';'):
        key=brand+' '+model
        if key in catalog:
            catalog[key]['requested_ranges'].append(int(tier));continue
        catalog[key]=dict(brand=brand,family=model,requested_ranges=[int(tier)],
                          requested_aliases_or_variants=ALIASES.get(key,[]),
                          checkpoint_22_class=key in supported,
                          training_counts=data['counts'].get(key),
                          status='trained_family_candidate_2.2' if key in supported else 'not_trained',
                          verified_good_real_phone_accuracy=False)
out=ROOT/'data/catalogue';out.mkdir(exist_ok=True)
report=dict(origin='User-requested target scope, 25 October 2026',
            names_are_not_verified_vehicle_specifications=True,
            policy='Aliases/trim claims require source review before import. No price, city prevalence, drivetrain, exact trim or perfect-recognition claims. A catalogue entry never supplies an inference answer.',
            entries=list(catalog.values()))
(out/'ev_target_catalog.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
covered=[k for k,r in catalog.items() if r['checkpoint_22_class']]
lines=['ЦЕЛЕВОЕ ПОКРЫТИЕ ПО СПИСКУ ПОЛЬЗОВАТЕЛЯ',
       f'Запрошено записей семейств: {len(catalog)}. Есть в классификаторе 2.2: {len(covered)}.',
       'Это проверка наличия класса, не подтверждение хорошей точности на реальном телефоне.',
       'Наличие варианта двигателя/комплектации в описании не означает его различимость на фото.',
       'Названия и алиасы из запроса требуют проверки по источникам до включения в обучение.',
       '', 'ЕСТЬ КЛАСС В КАНДИДАТЕ 2.2:']+covered+['','ЕЩЁ НЕ ОБУЧЕНЫ:']+[k for k in catalog if k not in supported]
(ROOT/'TARGET_EV_COVERAGE_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8-sig')
print(json.dumps(dict(requested=len(catalog),present=len(covered),missing=len(catalog)-len(covered))))
