"""Report reviewed training diagnostics separately from heldout public results."""
import csv,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
def iou(a,b):
    area=lambda z:max(0,z[2]-z[0])*max(0,z[3]-z[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])]);return inter/max(1,area(a)+area(b)-inter)
def main():
    out=ROOT/'outputs/mashinki-v29';base=json.loads((out/'summary.json').read_text());cand=json.loads((out/'candidate_summary.json').read_text());assert base['complete'] and cand['complete']
    annotations=[r for r in torch.load(ROOT/'cache/owner_domain_v29/data.pt',weights_only=True)['rows'] if r.get('domain_v29')]
    proposals=torch.load(ROOT/'models/owner_domain_v29/proposals.pt',weights_only=True)['rows'];approved=json.loads((ROOT/'models/owner_domain_v29/review_approval.json').read_text());labels=json.loads((ROOT/'models/owner_domain_v29/annotations.json').read_text());labels=[r for i,r in enumerate(labels) if str(i) not in approved['exclusions']]
    assert len(labels)==len(annotations)
    lookup={Path(r['source']).name:r for r in base['records']};newlookup={Path(r['source']).name:r for r in cand['records']};records=[]
    for label in labels:
        rec=dict(file=Path(label['source']).name,truth=label['brand'],truth_colour=label['colour'])
        for name,bank in [('baseline',lookup),('candidate',newlookup)]:
            cars=bank[rec['file']]['cars'];match=max(cars,key=lambda r:iou(r['box'],label['box']),default=None)
            if match is None or iou(match['box'],label['box'])<.3:rec[name]=dict(matched=False,correct=False,colour_correct=False,candidate_correct=False);continue
            evidence=match.get('make_evidence') or {};rec[name]=dict(matched=True,brand=match['brand'],candidate=evidence.get('candidate'),colour=match['colour'],confidence=evidence.get('confidence'),correct=match['brand']==label['brand'],candidate_correct=evidence.get('candidate')==label['brand'],colour_correct=match['colour']==label['colour'])
        records.append(rec)
    stats={k:{m:sum(bool(r[k][m]) for r in records) for m in ('matched','correct','candidate_correct','colour_correct')} for k in ('baseline','candidate')}
    report=dict(photos=len(cand['records']),proposals=sum(len(r['cars']) for r in cand['records']),reviewed_train_examples=len(records),counts=stats,records=records,scope='All scored own examples are TRAIN/development, not independent phone accuracy. All99 photographs processed but not all391 proposals manually labelled.')
    (out/'reviewed_training_comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    text='Проверка CarVision2.9 на D:\\машинки\n\nОбработаны все99 фотографий. Найденные контуры — предложения детектора, не проверенное число машин.\n'
    text+=f"На {len(records)} вручную просмотренных УЧЕБНЫХ фрагментах:\n"
    for name,s in stats.items():text+=f"{name}: подтверждена верная марка {s['correct']}/{len(records)}, верная верхняя гипотеза {s['candidate_correct']}/{len(records)}, цвет {s['colour_correct']}/{len(records)}.\n"
    text+='Это повторная проверка обучающих данных, не независимая точность.\n\n'
    for r in records:text+=f"{r['file']}: ожидается {r['truth']} {r['truth_colour']}; было {r['baseline'].get('brand','нет контура')}; стало {r['candidate'].get('brand','нет контура')} (гипотеза {r['candidate'].get('candidate','нет')}); цвет {r['candidate'].get('colour','нет')}.\n"
    text+='\nОткрытые проблемы: неверный цвет на части кадров, ложные контуры салона, unknown для ряда марок, ограниченная голова конкретной модели. Все325 моделей не обучены.\nОтдельный публичный тест937: top1 марки81.54% ->83.14%; искусственно ухудшенные71.29% ->76.95%. Нужен новый слепой тест телефонных фото.\n'
    (out/'RESULTS_RU.txt').write_text(text,encoding='utf-8-sig');print(json.dumps(dict(counts=stats,reviewed=len(records),all_photos=len(cand['records']))),flush=True)
if __name__=='__main__':main()
